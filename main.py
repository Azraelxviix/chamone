#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# main.py - Version 16.6.4 (Updated for Resilience, SDK Alignment, Secret Manager Fix, and SyntaxError Fix)
#test build cloudrun
import json
import logging
import os
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import functions_framework
import numpy as np
import psycopg
from google.api_core import exceptions as api_exceptions
from google.api_core.client_options import ClientOptions
from google.cloud import aiplatform, documentai as docai_v1, firestore, secretmanager, storage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from psycopg import sql
from sentence_transformers import SentenceTransformer
from cloudevents.http import CloudEvent

from google import genai
from google.genai import types

SCRIPT_VERSION = "16.6.4"
LOCK_LEASE_MINUTES = 20

os.environ["PROJECT_ID"] = "thebestever"
os.environ["VERTEX_AI_LOCATION"] = "us-central1"
os.environ["DOCAI_LOCATION"] = "us"
os.environ["INPUT_BUCKET"] = "knowledge-base-docs-thebestever"
os.environ["LOCK_BUCKET"] = "kblock"
os.environ["JSON_BUCKET"] = "kbjson"
os.environ["LOG_BUCKET"] = "kblogs"
os.environ["INSTRUCTIONS_BUCKET"] = "kbinfo"
os.environ["DOCAI_PROCESSOR_ID"] = "6e8f23fa5796a22b"
os.environ["INDEX_ENDPOINT_ID"] = "556724518584844288"
os.environ["DEPLOYED_INDEX_ID"] = "analysis_1756251260790"
os.environ["MASTER_INSTRUCTIONS_FILE"] = "master_instructions.txt"
os.environ["DB_USER"] = "retrieval-service"
os.environ["DB_NAME"] = "postgres"
os.environ["INSTANCE_CONNECTION_NAME"] = "thebestever:us-central1:genai-rag-db-6bdb68ec"
os.environ["ANALYSIS_MODEL"] = "models/gemini-2.5-pro"
os.environ["CLEANING_MODEL"] = "models/gemini-2.5-flash"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log_buffer = []


def get_firestore_client(project_id: str, database: str = "(default)") -> firestore.Client:
    return firestore.Client(project=project_id, database=database)


def get_storage_client() -> storage.Client:
    return storage.Client()


def get_docai_client(location: str) -> docai_v1.DocumentProcessorServiceClient:
    client_options = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    return docai_v1.DocumentProcessorServiceClient(client_options=client_options)


def get_gemini_api_key() -> Optional[str]:
    """Attempt to retrieve Gemini API key from Secret Manager, return None if not found."""
    try:
        client = secretmanager.SecretManagerServiceClient()
        project_id = os.environ["PROJECT_ID"]
        secret_name = "gemini_api"  # Using the name from your screenshot
        secret_version = "latest"
        name = f"projects/{project_id}/secrets/{secret_name}/versions/{secret_version}"
        response = client.access_secret_version(name=name)
        logging.info("Successfully retrieved Gemini API key from Secret Manager.")
        return response.payload.data.decode("UTF-8")
    except api_exceptions.NotFound:
        logging.warning("Gemini API key secret not found in Secret Manager. Falling back to ADC.")
        return None
    except Exception as e:
        logging.error(f"Failed to retrieve Gemini API key from Secret Manager: {e}")
        return None


def initialize_genai_client():
    """Initialize genai.Client with ADC as primary, falling back to Secret Manager."""
    api_key = get_gemini_api_key()
    logging.info("Initializing Google GenAI client with API version 'v1'.")
    try:
        if api_key:
            return genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(api_version='v1')
            )
        logging.info("No API key found, using Application Default Credentials.")
        return genai.Client(
            http_options=types.HttpOptions(api_version='v1')
        )
    except Exception as e:
        logging.critical(f"CRITICAL: Failed to initialize global clients: {e}")
        raise ValueError("Could not initialize Gemini client.")


def get_db_connection():
    try:
        client = secretmanager.SecretManagerServiceClient()
        project_id = os.environ["PROJECT_ID"]
        secret_name = "db_retrieval_pass"  # UPDATED LINE
        secret_version = "latest"
        name = f"projects/{project_id}/secrets/{secret_name}/versions/{secret_version}"
        response = client.access_secret_version(name=name)
        password = response.payload.data.decode("UTF-8")
        db_user = os.environ["DB_USER"]
        db_name = os.environ["DB_NAME"]
        db_host = "127.0.0.1"
        db_port = "5432"
        conn_str = f"dbname={db_name} user={db_user} password={password} host={db_host} port={db_port}"
        logging.info("Attempting to connect to the database via Cloud SQL Auth Proxy (TCP)...")
        conn = psycopg.connect(conn_str)
        logging.info("Database connection established successfully.")
        return conn
    except Exception as e:
        logging.error(f"CRITICAL: Database connection failed: {e}")
        raise


def _log(step: int, total_steps: int, message: str, is_error: bool = False, is_final: bool = False, level: str = "INFO") -> None:
    global log_buffer
    prefix = (f"Step {step}/{total_steps}: FAILED" if is_error else
              f"Pipeline Complete!" if is_final else
              f"Step {step}/{total_steps}: SUCCESS")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_line = f"[{timestamp}] {prefix} - {message}"
    log_buffer.append(log_line)
    if is_error or level.upper() == "ERROR":
        logging.error(log_line)
    elif level.upper() == "WARNING":
        logging.warning(log_line)
    else:
        logging.info(log_line)
    print(log_line)


def sanitize_filename(filename: str) -> str:
    return re.sub(r'[^a-zA-Z0-9.-]', '_', filename)


EMBEDDING_MODEL = None


def _get_embedding_model():
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        _log(0, 0, "LAZY LOADING: Initializing SentenceTransformer model...")
        EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        _log(0, 0, "LAZY LOADING: SentenceTransformer model loaded successfully.")
    return EMBEDDING_MODEL


def check_and_reserve_document(case_id: str, case_number: str, document_name: str, db_conn) -> Optional[int]:
    with db_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (case_id, case_number, original_filename, processed_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (original_filename) DO NOTHING
            RETURNING id;
            """,
            (case_id, case_number, document_name)
        )
        result = cur.fetchone()
        if result is None:
            logging.warning(f"Document '{document_name}' already exists. Skipping reservation.")
            return None
        db_conn.commit()
        return result[0]


def update_document_with_content(document_id: int, document_text: str, chunk_embeddings: Optional[np.ndarray], db_conn) -> None:
    with db_conn.cursor() as cur:
        embedding = None
        if chunk_embeddings is not None and len(chunk_embeddings) > 0:
            embedding = np.mean(chunk_embeddings, axis=0).tolist()
        cur.execute(
            """
            UPDATE documents
            SET full_text = %s, full_text_embedding = %s, processed_at = NOW()
            WHERE id = %s;
            """,
            (document_text, embedding, document_id)
        )
        db_conn.commit()


def index_document_chunks(document_id: int, chunks: List[str], embeddings: np.ndarray, db_conn) -> None:
    with db_conn.cursor() as cur:
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                """
                INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (document_id, chunk_index) DO UPDATE
                SET chunk_text = EXCLUDED.chunk_text, embedding = EXCLUDED.embedding, updated_at = NOW();
                """,
                (document_id, i, chunk, embedding.tolist())
            )
        db_conn.commit()
    _log(4, 7, f"Successfully inserted/updated {len(chunks)} chunks into Cloud SQL for document_id {document_id}.")


def _call_generative_model(
    task_name: str,
    client: genai.Client,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    is_json: bool = False,
    response_schema: Optional[Dict] = None,
) -> Optional[str]:
    logging.info(f"Calling generative model for: {task_name}")
    try:
        generation_config_params = {
            "temperature": 0.1,
            "top_p": 0.95,
            "max_output_tokens": 8192,
        }
        if is_json and response_schema:
            generation_config_params["response_mime_type"] = "application/json"
            if response_schema:
                generation_config_params["response_schema"] = response_schema
        contents = [system_prompt, user_prompt]
        response = client.generate_content(
            model=model_name,
            contents=contents,
            generation_config=generation_config_params
        )
        return response.text
    except api_exceptions.NotFound as e:
        logging.error(f"Model '{model_name}' not found. Details: {e}")
        return None
    except api_exceptions.InvalidArgument as e:
        logging.error(f"Invalid argument in request to model '{model_name}'. Details: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during the API call: {e}")
        return None


def _get_context_from_rag(query_embedding: Optional[np.ndarray], db_conn, project_id: str, vertex_ai_location: str, index_endpoint_id: str, deployed_index_id: str) -> str:
    if query_embedding is None or query_embedding.size == 0:
        return "No relevant historical context found (Embedding missing)."
    _log(0, 0, "Retrieving context from RAG (Vertex AI Vector Search)...")
    try:
        query_vec = np.mean(query_embedding, axis=0).tolist() if query_embedding.ndim > 1 else query_embedding.tolist()
        index_endpoint = aiplatform.MatchingEngineIndexEndpoint(f"projects/{project_id}/locations/{vertex_ai_location}/indexEndpoints/{index_endpoint_id}")
        response = index_endpoint.find_neighbors(
            deployed_index_id=deployed_index_id,
            queries=[query_vec],
            num_neighbors=5
        )
        if not response or not response[0]:
            return "No relevant historical context found in Vector Search."
        with db_conn.cursor() as cur:
            neighbor_ids = [int(neighbor.id) for neighbor in response[0]]
            if not neighbor_ids:
                return "No relevant historical context found in Vector Search."
            query = sql.SQL("SELECT chunk_text, (embedding <=> %s) AS distance FROM document_chunks WHERE id = ANY(%s) ORDER BY distance ASC")
            cur.execute(query, (str(query_vec), neighbor_ids))
            results = cur.fetchall()
            context = [f"- (Distance: {row[1]:.4f}): {row[0]}" for row in results]
            return "\n".join(context) if context else "No relevant historical context found."
    except Exception as e:
        logging.error(f"Error during RAG retrieval: {e}", exc_info=True)
        return "Error retrieving historical context."


def get_clean_document_text(input_file: str, mime_type: str, processor_id: str, temp_bucket: str, client: genai.Client) -> Optional[str]:
    try:
        docai_client = get_docai_client(os.environ["DOCAI_LOCATION"])
        processor_name = docai_client.processor_path(os.environ["PROJECT_ID"], os.environ["DOCAI_LOCATION"], processor_id)
        output_gcs_uri = f"gs://{temp_bucket}/ocr/{input_file.split('gs://')[-1]}/"
        operation = docai_client.batch_process_documents(
            request=docai_v1.BatchProcessRequest(
                name=processor_name,
                input_documents=docai_v1.BatchDocumentsInputConfig(
                    gcs_documents=docai_v1.GcsDocuments(
                        documents=[docai_v1.GcsDocument(gcs_uri=input_file, mime_type=mime_type)],
                    ),
                ),
                document_output_config=docai_v1.DocumentOutputConfig(
                    gcs_output_config=docai_v1.DocumentOutputConfig.GcsOutputConfig(
                        gcs_uri=output_gcs_uri,
                        field_mask="text"
                    ),
                ),
            ),
        )
        operation.result(timeout=1800)
        storage_client = get_storage_client()
        metadata = docai_v1.BatchProcessMetadata(operation.metadata)
        if not metadata.individual_process_statuses:
            logging.error("No individual process statuses returned from Document AI.")
            return None
        output_gcs_path = metadata.individual_process_statuses[0].output_gcs_destination
        bucket_name, prefix = output_gcs_path.removeprefix("gs://").split("/", 1)
        output_blobs = list(storage_client.list_blobs(bucket_name, prefix=prefix))
        raw_text = "".join(
            docai_v1.Document.from_json(blob.download_as_bytes(), ignore_unknown_fields=True).text
            for blob in output_blobs if blob.name.endswith(".json")
        )
        for blob in output_blobs:
            blob.delete()
        return _ai_text_restoration(raw_text, client) if raw_text.strip() else None
    except Exception as e:
        logging.error(f"Generic OCR failure: {e}")
        return None


def _ai_text_restoration(text: str, client: genai.Client) -> str:
    return _call_generative_model(
        "AI Text Restoration",
        client=client,
        model_name=os.environ["CLEANING_MODEL"],
        system_prompt=(
            "The following text was extracted via OCR and may contain errors."
            "Your task is to clean and restore it. Do not add or remove information, only correct errors. "
            "Preserve paragraph structure."
        ),
        user_prompt=f"--- ORIGINAL TEXT ---\n{text}\n--- RESTORED TEXT ---",
        is_json=False
    ) or text


def generate_legal_analysis(
    document_text: str,
    document_name: str,
    db_conn,
    chunk_embeddings: Optional[np.ndarray],
    storage_client,
    instructions_bucket_name,
    master_instructions_file,
    analysis_model_name,
    client: genai.Client,
    project_id,
    vertex_ai_location,
    index_endpoint_id,
    deployed_index_id
) -> Optional[dict]:
    _log(5, 7, "Generating legal analysis...")
    try:
        instructions_blob = storage_client.bucket(instructions_bucket_name).blob(master_instructions_file)
        master_instructions = instructions_blob.download_as_text()
        logging.info("Successfully loaded master instructions from GCS.")
    except Exception as e:
        logging.warning(f"Could not load master instructions from GCS: {e}. Using default.")
        master_instructions = "You are an expert legal analyst."
    historical_context = _get_context_from_rag(chunk_embeddings, db_conn, project_id, vertex_ai_location, index_endpoint_id, deployed_index_id)
    analysis_schema = {
        "type": "object",
        "properties": {
            "filing_details": {
                "type": "object",
                "properties": {
                    "case_number": {"type": "string"},
                    "case_micro_id": {"type": "string", "enum": ["AS", "AM", "DG", "AU", "LA", "MH", "HK", "MF", "UNKNOWN"]},
                    "court": {"type": "string"},
                    "parties": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "party_codified_id": {"type": "string", "enum": [
                                    "AGH", "CLD", "LMH", "PAIS", "KIT", "TEC", "MBA", "FPMG", "AKD", "DSM",
                                    "MDM", "CRM", "SWM", "BGM", "LHM", "KTX", "TTA", "JBA", "CRA", "DWA",
                                    "SCC", "BBP", "BRD", "KJM", "TLN", "DJC", "DAD", "CDM", "ACM", "SRM",
                                    "DBM", "JDP", "HSC", "VHC", "SCD", "BCD", "MCD", "USA", "DVA", "DHS",
                                    "UCR", "UCV", "UCPD", "ACI", "SRD", "DRS", "HBT", "CHP", "FEMA", "JCD",
                                    "UNKNOWN"
                                ]},
                                "role": {"type": "string", "enum": ["Plaintiff", "Defendant", "Other"]}
                            },
                            "required": ["name", "role", "party_codified_id"]
                        }
                    }
                },
                "required": ["case_number", "court", "parties", "case_micro_id"]
            },
            "key_events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "event_date": {"type": "string", "format": "date"},
                        "event_description": {"type": "string"},
                        "page_reference": {"type": "integer"}
                    },
                    "required": ["event_date", "event_description", "page_reference"]
                }
            },
            "legal_criticism": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["filing_details", "key_events", "legal_criticism"]
    }
    system_prompt_lines = [
        f"# PERSONA",
        f"{master_instructions}",
        "",
        f"# TASK",
        "Critically analyze the legal document provided. Identify key details, dates, and potential legal infractions.",
        "Structure your response as a single, valid JSON object.",
        "",
        f"# MASTER ID LISTS (Use these for categorization)",
        f"Case Micro IDs: {', '.join(['AS', 'AM', 'DG', 'AU', 'LA', 'MH', 'HK', 'MF', 'UNKNOWN'])}",
        f"Party Codified IDs: {', '.join(['AGH', 'CLD', 'LMH', 'PAIS', 'KIT', 'TEC', 'MBA', 'FPMG', 'AKD', 'DSM', 'MDM', 'CRM', 'SWM', 'BGM', 'LHM', 'KTX', 'TTA', 'JBA', 'CRA', 'DWA', 'SCC', 'BBP', 'BRD', 'KJM', 'TLN', 'DJC', 'DAD', 'CDM', 'ACM', 'SRM', 'DBM', 'JDP', 'HSC', 'VHC', 'SCD', 'BCD', 'MCD', 'USA', 'DVA', 'DHS', 'UCR', 'UCV', 'UCPD', 'ACI', 'SRD', 'DRS', 'HBT', 'CHP', 'FEMA', 'JCD', 'UNKNOWN'])}",
        "",
        f"# FORMAT",
        "Return a JSON object matching the required schema. CRITICAL: You MUST use the Master ID Lists to populate 'case_micro_id' and 'party_codified_id'.",
        "If an ID cannot be determined, use 'UNKNOWN'."
    ]
    system_prompt = "\n".join(system_prompt_lines)
    user_prompt_lines = [
        f"# CONTEXT",
        f"--- RELEVANT CASE HISTORY ---",
        f"{historical_context}",
        f"--- END HISTORY ---",
        "",
        f"--- NEW DOCUMENT TO ANALYZE ---",
        f"{document_text}",
        f"--- END DOCUMENT ---"
    ]
    user_prompt = "\n".join(user_prompt_lines)
    analysis_text = _call_generative_model(
        "Legal Analysis", client, analysis_model_name, system_prompt, user_prompt, is_json=True, response_schema=analysis_schema
    )
    if not analysis_text:
        return None
    try:
        match = re.search(r"```json(.*?)```", analysis_text, re.DOTALL)
        if match:
            json_text = match.group(1)
        else:
            json_text = analysis_text
        analysis_json = json.loads(json_text)
        return analysis_json if "legal_criticism" in analysis_json else None
    except json.JSONDecodeError:
        logging.error(f"Failed to decode model output into JSON: {analysis_text[:500]}...")
        raise ValueError("Model output was not valid JSON.")


def generate_criticism_timeline(analysis: dict) -> Optional[str]:
    criticisms = analysis.get("legal_criticism", [])
    events = analysis.get("key_events", [])
    if not criticisms and not events:
        return "No criticisms or key events identified."
    timeline = ["# Criticism and Event Timeline\n"]
    if events:
        timeline.append("## Key Events\n")
        sorted_events = sorted(events, key=lambda x: x.get("event_date", "0000-00-00"))
        for event in sorted_events:
            date = event.get("event_date", "N/A")
            desc = event.get("event_description", "N/A")
            page = event.get("page_reference", "N/A")
            timeline.append(f"- {date} (Page {page}): {desc}")
    if criticisms:
        timeline.append("\n## Legal Criticisms\n")
        for criticism in criticisms:
            timeline.append(f"- {criticism}")
    return "\n".join(timeline)


def _persist_analysis_hierarchical(
    filename: str,
    clean_text: Optional[str],
    analysis: Optional[dict],
    timeline: Optional[str],
    db_client,
    db_conn,
    chunk_embeddings: Optional[np.ndarray],
    analysis_model_name: str,
    cleaning_model_name: str,
    case_micro_id: str,
    case_number: str,
    document_id: Optional[int]
) -> None:
    doc_id = sanitize_filename(filename)
    if clean_text or analysis or timeline:
        logging.info(f"Persisting available data for '{filename}' (ID: {doc_id}) to Firestore...")
        doc_ref = db_client.collection("documents").document(doc_id)
        batch = db_client.batch()
        processing_status = "COMPLETE" if analysis else "PARTIAL"
        doc_data = {
            "source_document": filename,
            "processing_status": processing_status,
            "last_processed_utc": firestore.SERVER_TIMESTAMP,
            "ai_metadata": {
                "analysis_model": analysis_model_name,
                "cleaning_model": cleaning_model_name,
                "script_version": SCRIPT_VERSION
            }
        }
        if analysis:
            doc_data["filing_details"] = analysis.get("filing_details", {})
        batch.set(doc_ref, doc_data, merge=True)
        if clean_text:
            batch.set(doc_ref.collection("content").document("clean_text"), {"text": clean_text})
        if timeline:
            batch.set(doc_ref.collection("syntheses").document("criticism_timeline"), {"markdown_text": timeline})
        batch.commit()
        logging.info(f"Successfully persisted data to Firestore for '{doc_id}'. Status: {processing_status}")
    else:
        logging.warning(f"Skipping Firestore persistence for '{filename}' as no data (OCR or Analysis) was generated.")
    if clean_text and clean_text.strip():
        try:
            if document_id is None:
                document_id = check_and_reserve_document(case_micro_id, case_number, filename, db_conn)
            if document_id:
                update_document_with_content(document_id, clean_text, chunk_embeddings, db_conn)
                if chunk_embeddings is not None and len(chunk_embeddings) > 0:
                    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200, length_function=len).split_text(clean_text)
                    index_document_chunks(document_id, chunks, chunk_embeddings, db_conn)
                logging.info(f"Stored document and embedding in Cloud SQL for '{doc_id}'.")
            else:
                logging.warning(f"Skipping Cloud SQL persistence for '{filename}' due to duplicate document.")
        except Exception as e:
            logging.error(f"CRITICAL: Failed to persist data to Cloud SQL for '{doc_id}': {e}")
            if db_conn:
                db_conn.rollback()
            raise
    else:
        logging.info(f"Skipping Cloud SQL persistence for '{filename}' as clean text is missing.")


def _handle_processing_with_lease_lock(filename: str, mime_type: str):
    storage_client = get_storage_client()
    lock_bucket_name = os.environ.get("LOCK_BUCKET")
    if not lock_bucket_name:
        logging.error("FATAL: LOCK_BUCKET environment variable not set.")
        return
    lock_file_name = f"{sanitize_filename(filename)}.lock"
    bucket = storage_client.bucket(lock_bucket_name)
    blob = bucket.blob(lock_file_name)
    lease_acquired = False
    try:
        blob.reload()
        lock_content = blob.download_as_text()
        lock_timestamp = datetime.fromisoformat(lock_content)
        current_time = datetime.now(timezone.utc)
        elapsed_time = current_time - lock_timestamp
        if elapsed_time < timedelta(minutes=LOCK_LEASE_MINUTES):
            logging.warning(f"Lease for '{filename}' is active (acquired {elapsed_time.total_seconds() / 60:.2f} mins ago). Skipping.")
            return
        else:
            logging.info(f"Stale lease for '{filename}' found. Attempting to acquire.")
            stale_generation = blob.generation
            new_lease_time = datetime.now(timezone.utc).isoformat()
            blob.upload_from_string(new_lease_time, if_generation_match=stale_generation)
            lease_acquired = True
    except api_exceptions.NotFound:
        logging.info(f"No active lease for '{filename}'. Attempting to acquire.")
        new_lease_time = datetime.now(timezone.utc).isoformat()
        blob.upload_from_string(new_lease_time, if_generation_match=0)
        lease_acquired = True
    except api_exceptions.PreconditionFailed:
        logging.warning(f"Failed to acquire lease for '{filename}'; another process won the race. Skipping.")
        return
    if lease_acquired:
        logging.info(f"Lease acquired for '{filename}'. Starting processing.")
        try:
            process_document_pipeline(filename=filename, mime_type=mime_type)
        finally:
            try:
                blob.reload()
                blob.delete()
                logging.info(f"Lease for '{filename}' released.")
            except Exception as e:
                logging.error(f"CRITICAL: Failed to release lease for '{filename}': {e}")


@functions_framework.cloud_event
def on_cloud_event(event: CloudEvent) -> None:
    try:
        filename = event.data["name"]
        mime_type = event.data["contentType"]
        logging.info(f"Received CloudEvent for file: {filename} ({mime_type})")
        _handle_processing_with_lease_lock(filename=filename, mime_type=mime_type)
    except Exception as e:
        logging.critical(f"CRITICAL: Failed to process CloudEvent: {e}", exc_info=True)
        return


def process_document_pipeline(filename: str, mime_type: str):
    global log_buffer
    log_buffer = []
    TOTAL_STEPS = 7
    try:
        project_id = os.environ["PROJECT_ID"]
        docai_processor_id = os.environ["DOCAI_PROCESSOR_ID"]
        input_bucket_name = os.environ["INPUT_BUCKET"]
        lock_bucket_name = os.environ["LOCK_BUCKET"]
        json_bucket_name = os.environ["JSON_BUCKET"]
        log_bucket_name = os.environ["LOG_BUCKET"]
        instructions_bucket_name = os.environ["INSTRUCTIONS_BUCKET"]
        docai_location = os.environ["DOCAI_LOCATION"]
        vertex_ai_location = os.environ["VERTEX_AI_LOCATION"]
        index_endpoint_id = os.environ["INDEX_ENDPOINT_ID"]
        deployed_index_id = os.environ["DEPLOYED_INDEX_ID"]
        master_instructions_file = os.environ["MASTER_INSTRUCTIONS_FILE"]
        analysis_model_name = os.environ["ANALYSIS_MODEL"]
        cleaning_model_name = os.environ["CLEANING_MODEL"]
        firestore_db = get_firestore_client(project_id)
        storage_client = get_storage_client()
        client = initialize_genai_client()
        aiplatform.init(project=project_id, location=vertex_ai_location)
        db_conn = None
        try:
            db_conn = get_db_connection()
            try:
                extracted_case_micro_id = filename.split('_')[1]
            except IndexError:
                logging.error(f"Could not extract case_micro_id from filename: {filename}")
                raise ValueError("Invalid filename format for case_micro_id extraction.")
            with db_conn.cursor() as cur:
                cur.execute("SELECT id FROM cases WHERE case_micro_id = %s", (extracted_case_micro_id,))
                result = cur.fetchone()
                if not result:
                    logging.error(f"Case with micro_id '{extracted_case_micro_id}' not found in cases table.")
                    raise ValueError("Case not found.")
                case_id = result[0]
            _log(0, TOTAL_STEPS, f"Reserving document entry for {filename}")
            document_id = check_and_reserve_document(case_id, "UNKNOWN", filename, db_conn)
            if document_id is None:
                _log(0, TOTAL_STEPS, f"Document {filename} already exists. Skipping processing.")
                return
            input_gcs_uri = f"gs://{input_bucket_name}/{filename}"
            _log(1, TOTAL_STEPS, f"Getting document text for {filename}")
            clean_text = get_clean_document_text(input_gcs_uri, mime_type, docai_processor_id, json_bucket_name, client)
            if not clean_text:
                raise ValueError("OCR process resulted in no clean text.")
            _log(2, TOTAL_STEPS, f"Extracted and cleaned text from {filename}")
            _log(3, TOTAL_STEPS, f"Generating embeddings for {filename}")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200, length_function=len)
            chunks = text_splitter.split_text(clean_text)
            embeddings = None
            if chunks:
                model = _get_embedding_model()
                embeddings = model.encode(chunks, batch_size=16)
            _log(3, TOTAL_STEPS, f"Generated {len(embeddings) if embeddings is not None else 0} embeddings.")
            analysis = generate_legal_analysis(
                clean_text, filename, db_conn, embeddings, storage_client,
                instructions_bucket_name, master_instructions_file, analysis_model_name, client,
                project_id, vertex_ai_location, index_endpoint_id, deployed_index_id
            )
            timeline = generate_criticism_timeline(analysis) if analysis else None
            _log(4, TOTAL_STEPS, f"Persisting analysis for {filename}")
            _persist_analysis_hierarchical(
                filename, clean_text, analysis, timeline, firestore_db, db_conn,
                embeddings, analysis_model_name, cleaning_model_name,
                extracted_case_micro_id, "UNKNOWN", document_id
            )
            _log(7, TOTAL_STEPS, "Document processing complete.", is_final=True)
        except Exception as e:
            if db_conn:
                db_conn.rollback()
            _log(0, TOTAL_STEPS, f"Pipeline failed: {e}", is_error=True)
            raise
        finally:
            if db_conn:
                db_conn.close()
    except Exception as e:
        logging.critical(f"CRITICAL: Failed to initialize global clients: {e}")
