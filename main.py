#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ╔═╗╔╗╔╔═╗╦╔═╗╔╗╔╦╔═╗╦═╗
# ║ ╦╠╩╣╠═╣║║ ╦║║║║║ ║╠╦╝
# ╚═╝╚═╝╩ ╩╩╚═╝╝╚╝╩╚═╝╩╚═
#
# main.py - Version 17.1.0
# Maintained by Gemini & gr0k
# Last updated by 🚀 Gemini 🚀

#### GOLDEN RULE FOR THIS SCRIPT ####
# WE ARE NOT USING THE GOOGLE-GENERATIVEAI SDK.
# ALL GEMINI INTERACTIONS ARE HANDLED THROUGH THE ROBUST GOOGLE-CLOUD-AIPLATFORM SDK.
#####################################

import os
import re
import json
import logging
import datetime
from typing import Optional, List, Dict, Any

# --- Cloud & Framework Imports ---
import functions_framework
from cloudevents.http import CloudEvent
from google.cloud import aiplatform, documentai, storage
from google.api_core import exceptions as google_exceptions

# --- Database Imports ---
import psycopg
from psycopg_pool import ConnectionPool

# --- Text Processing Imports ---
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==================================================================================================
#    UNIT 0: CONFIGURATION & INITIALIZATION 🔥
# ==================================================================================================

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Environment Variable Loading ---
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("VERTEX_AI_LOCATION")
INPUT_BUCKET_NAME = os.environ.get("INPUT_BUCKET")
LOCK_BUCKET_NAME = os.environ.get("LOCK_BUCKET")
CLEANED_TEXT_BUCKET_NAME = "cleaned-text-thebestever"
ANALYSIS_JSON_BUCKET_NAME = "analysis-json-thebestever"
DOCAI_PROCESSOR_ID = os.environ.get("DOCAI_PROCESSOR_ID")
DOCAI_LOCATION = os.environ.get("DOCAI_LOCATION")
CLEANING_MODEL = os.environ.get("CLEANING_MODEL")
ANALYSIS_MODEL = os.environ.get("ANALYSIS_MODEL")
EMBEDDING_MODEL_NAME = "text-embedding-004"
DB_USER = os.environ.get("DB_USER")
DB_NAME = os.environ.get("DB_NAME")
DB_PASSWORD_SECRET = "DB_PASSWORD"

# --- Global Client Initialization ---
storage_client = storage.Client()
aiplatform.init(project=PROJECT_ID, location=REGION)
docai_client = documentai.DocumentProcessorServiceClient()
db_pool: Optional[ConnectionPool] = None

def get_db_password() -> str:
    """Retrieves the database password from Secret Manager."""
    from google.cloud import secretmanager
    client = secretmanager.SecretManagerServiceClient()
    secret_name = f"projects/{PROJECT_ID}/secrets/{DB_PASSWORD_SECRET}/versions/latest"
    try:
        response = client.access_secret_version(request={"name": secret_name})
        return response.payload.data.decode("UTF-8")
    except google_exceptions.NotFound:
        logging.critical(f"🛑 FATAL: Secret '{DB_PASSWORD_SECRET}' not found.")
        raise

def initialize_db_pool():
    """Initializes the global database connection pool."""
    global db_pool
    if db_pool is None:
        logging.info("🔥 Initializing database connection pool...")
        try:
            conninfo = (
                f"dbname={DB_NAME} user={DB_USER} "
                f"password={get_db_password()} "
                f"host=127.0.0.1 port=5432"
            )
            db_pool = ConnectionPool(conninfo=conninfo, min_size=1, max_size=5, open=True)
            logging.info("✅ Database connection pool initialized successfully.")
        except Exception as e:
            logging.critical(f"🛑 CRITICAL: Could not initialize database pool: {e}")
            raise

# --- Jubilee Script: Unlock Leases on New Deployment ---
DEPLOYMENT_ID = os.environ.get("DEPLOYMENT_ID")
if DEPLOYMENT_ID:
    logging.info(f"🚀 New deployment detected (ID: {DEPLOYMENT_ID}). Initiating Jubilee...")
    try:
        lock_bucket = storage_client.bucket(LOCK_BUCKET_NAME)
        blobs = list(lock_bucket.list_blobs())
        if blobs:
            lock_bucket.delete_blobs(blobs)
            logging.info(f"✅ JUBILEE COMPLETE: Deleted {len(blobs)} old lock files.")
        else:
            logging.info("✅ JUBILEE COMPLETE: No old lock files found.")
    except Exception as e:
        logging.critical(f"🛑 JUBILEE FAILED: Could not clear lock files: {e}")

# ==================================================================================================
#    MAIN CLOUD FUNCTION ENTRYPOINT
# ==================================================================================================

@functions_framework.cloud_event
def on_cloud_event(cloud_event: CloudEvent) -> None:
    """Main entry point, triggered by a GCS file upload."""
    data = cloud_event.data
    bucket_name = data.get("bucket")
    file_name = data.get("name")

    if not all([bucket_name, file_name]):
        logging.error("🛑 Received malformed CloudEvent, missing bucket or file name.")
        return

    logging.info(f"🎬 Received CloudEvent for file: gs://{bucket_name}/{file_name}")
    
    # Initialize the pool on the first invocation if it hasn't been already
    if db_pool is None:
        initialize_db_pool()
        
    process_document_pipeline(bucket_name, file_name)

# ==================================================================================================
#    MASTER PIPELINE CONTROLLER
# ==================================================================================================

def process_document_pipeline(bucket_name: str, file_name: str) -> None:
    """Orchestrates the entire document processing workflow, unit by unit."""
    pipeline_state = { "file_name": file_name, "bucket_name": bucket_name, "lease_acquired": False }
    final_status = "SUCCESS"
    
    try:
        pipeline_state = unit_1_acquire_lease(pipeline_state)
        
        if pipeline_state.get("lease_acquired"):
            pipeline_state = unit_2_run_ocr(pipeline_state)
            pipeline_state = unit_3_clean_text(pipeline_state)
            pipeline_state = unit_4_archive_cleaned_text(pipeline_state)
            pipeline_state = unit_5_analyze_with_gemini(pipeline_state)
            pipeline_state = unit_6_archive_analysis(pipeline_state)
            pipeline_state = unit_7_generate_embeddings(pipeline_state)
            pipeline_state = unit_8_update_database(pipeline_state)
            
            if any("FAILED" in str(v) for k, v in pipeline_state.items() if "status" in k):
                final_status = "PARTIAL_FAILURE"

    except Exception as e:
        logging.critical(f"🛑 PIPELINE HALTED due to unrecoverable error: {e}", exc_info=True)
        final_status = "CRITICAL_FAILURE"
        update_document_status(pipeline_state.get("document_id"), final_status)

    finally:
        if pipeline_state.get("lease_acquired"):
            unit_9_release_lease(pipeline_state)
        logging.info(f"🏁 Pipeline finished for '{file_name}' with final status: {final_status}")

# ==================================================================================================
#    UNITS OF WORK
# ==================================================================================================

def unit_1_acquire_lease(state: Dict[str, Any]) -> Dict[str, Any]:
    """Atomically acquires a lease and creates the initial database record."""
    logging.info("▶️ Unit 1: Document Leasing & Reservation")
    file_name = state["file_name"]
    lock_blob = storage_client.bucket(LOCK_BUCKET_NAME).blob(f"{file_name}.lock")

    try:
        lock_blob.upload_from_string(
            f"Leased at {datetime.datetime.now(datetime.UTC).isoformat()}",
            if_generation_match=0
        )
        state["lease_acquired"] = True
        logging.info(f"   Lease acquired for '{file_name}'.")

        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO documents (original_filename, status, processed_at) VALUES (%s, 'PROCESSING', NOW()) "
                    "ON CONFLICT (original_filename) DO UPDATE SET status = 'PROCESSING', processed_at = NOW() "
                    "RETURNING id;",
                    (file_name,)
                )
                result = cur.fetchone()
                state["document_id"] = result[0] if result else None
        
        if not state["document_id"]:
             raise Exception("Failed to create or retrieve document record ID.")
        logging.info(f"   Initial document record created with ID: {state['document_id']}.")
        logging.info("✅ Unit 1: Complete.")

    except google_exceptions.PreconditionFailed:
        logging.warning(f"⚠️ Lease for '{file_name}' is already held. Skipping.")
        state["lease_acquired"] = False
        
    except Exception as e:
        logging.error(f"🚨 Unit 1 FAILED: {e}", exc_info=True)
        state["lease_acquired"] = False
        raise
    return state

def unit_2_run_ocr(state: Dict[str, Any]) -> Dict[str, Any]:
    """Performs OCR using Document AI."""
    logging.info("▶️ Unit 2: Document AI OCR")
    try:
        gcs_uri = f"gs://{state['bucket_name']}/{state['file_name']}"
        resource_name = docai_client.processor_path(PROJECT_ID, DOCAI_LOCATION, DOCAI_PROCESSOR_ID)
        request = documentai.ProcessRequest(name=resource_name, gcs_document=documentai.GcsDocument(gcs_uri=gcs_uri, mime_type="application/pdf"))
        result = docai_client.process_document(request=request)
        state["full_text"] = result.document.text
        update_document_status(state["document_id"], "OCR_COMPLETE")
        logging.info(f"   Successfully extracted {len(state.get('full_text', ''))} characters.")
        logging.info("✅ Unit 2: Complete.")
    except Exception as e:
        logging.error(f"🚨 Unit 2 FAILED: {e}", exc_info=True)
        state["ocr_status"] = "FAILED"
        update_document_status(state["document_id"], "OCR_FAILED")
    return state

def unit_3_clean_text(state: Dict[str, Any]) -> Dict[str, Any]:
    """Cleans text using a Gemini model."""
    logging.info("▶️ Unit 3: Text Normalization & Cleaning")
    if not state.get("full_text"):
        logging.warning("   Skipping; no text from OCR.")
        return state
    try:
        model = aiplatform.GenerativeModel(CLEANING_MODEL)
        system_prompt = "You are an expert legal text cleaner. Your task is to process raw OCR text from a legal document. Correct OCR errors, normalize dates to YYYY-MM-DD format, standardize legal citations, and remove page/line numbers. Preserve the core legal text and structure. Return ONLY the cleaned text."
        response = model.generate_content([system_prompt, state["full_text"]])
        state["cleaned_text"] = response.text
        update_document_status(state["document_id"], "CLEANING_COMPLETE")
        logging.info("   Text cleaned successfully.")
        logging.info("✅ Unit 3: Complete.")
    except Exception as e:
        logging.error(f"🚨 Unit 3 FAILED: {e}", exc_info=True)
        state["cleaning_status"] = "FAILED"
        update_document_status(state["document_id"], "CLEANING_FAILED")
        state["cleaned_text"] = state.get("full_text") # Fallback
    return state

def unit_4_archive_cleaned_text(state: Dict[str, Any]) -> Dict[str, Any]:
    """Saves cleaned text to GCS."""
    logging.info("▶️ Unit 4: Archive Cleaned Text")
    if not state.get("cleaned_text"):
        logging.warning("   Skipping; no cleaned text available.")
        return state
    try:
        blob = storage_client.bucket(CLEANED_TEXT_BUCKET_NAME).blob(f"{state['file_name']}.txt")
        blob.upload_from_string(state["cleaned_text"], content_type="text/plain")
        logging.info(f"   Cleaned text archived to gs://{CLEANED_TEXT_BUCKET_NAME}/{blob.name}")
        logging.info("✅ Unit 4: Complete.")
    except Exception as e:
        logging.error(f"🚨 Unit 4 FAILED: {e}", exc_info=True)
        state["archive_text_status"] = "FAILED"
    return state
    
def unit_5_analyze_with_gemini(state: Dict[str, Any]) -> Dict[str, Any]:
    """Performs deep analysis on the text to extract structured data."""
    logging.info("▶️ Unit 5: Deep Analysis with Gemini")
    if not state.get("cleaned_text"):
        logging.warning("   Skipping; no cleaned text available.")
        return state
    try:
        model = aiplatform.GenerativeModel(ANALYSIS_MODEL)
        system_prompt = """You are a legal analysis expert. Analyze the provided legal document text and return ONLY a raw JSON object with the following schema: {"document_title": "string", "document_type": "string", "case_number": "string", "case_micro_id": "string (e.g., 'AS', 'AU', 'DG', or 'UNK')", "summary": "string", "key_entities": ["string"], "filing_date": "YYYY-MM-DD or null"}"""
        response = model.generate_content([system_prompt, state["cleaned_text"]])
        
        json_match = re.search(r'```json\n({.*})\n```', response.text, re.DOTALL)
        json_text = json_match.group(1) if json_match else response.text
        state["analysis_json"] = json.loads(json_text)
        
        update_document_status(state["document_id"], "ANALYSIS_COMPLETE")
        logging.info("   Document analysis complete.")
        logging.info("✅ Unit 5: Complete.")
    except Exception as e:
        logging.error(f"🚨 Unit 5 FAILED: {e}", exc_info=True)
        state["analysis_status"] = "FAILED"
        update_document_status(state["document_id"], "ANALYSIS_FAILED")
    return state

def unit_6_archive_analysis(state: Dict[str, Any]) -> Dict[str, Any]:
    """Saves the analysis JSON to GCS."""
    logging.info("▶️ Unit 6: Archive Analysis JSON")
    if not state.get("analysis_json"):
        logging.warning("   Skipping; no analysis JSON available.")
        return state
    try:
        blob = storage_client.bucket(ANALYSIS_JSON_BUCKET_NAME).blob(f"{state['file_name']}.json")
        blob.upload_from_string(json.dumps(state["analysis_json"], indent=2), content_type="application/json")
        logging.info(f"   Analysis JSON archived to gs://{ANALYSIS_JSON_BUCKET_NAME}/{blob.name}")
        logging.info("✅ Unit 6: Complete.")
    except Exception as e:
        logging.error(f"🚨 Unit 6 FAILED: {e}", exc_info=True)
        state["archive_json_status"] = "FAILED"
    return state

def unit_7_generate_embeddings(state: Dict[str, Any]) -> Dict[str, Any]:
    """Chunks text and generates embeddings using Vertex AI."""
    logging.info("▶️ Unit 7: Generate Embeddings")
    if not state.get("cleaned_text"):
        logging.warning("   Skipping; no cleaned text available.")
        return state
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(state["cleaned_text"])
        logging.info(f"   Text split into {len(chunks)} chunks.")
        
        from vertexai.language_models import TextEmbeddingModel
        model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
        response = model.get_embeddings(chunks)
        
        state["chunks"] = chunks
        state["embeddings"] = [embedding.values for embedding in response]
        
        update_document_status(state["document_id"], "EMBEDDING_COMPLETE")
        logging.info(f"   Successfully generated {len(state.get('embeddings', []))} embeddings.")
        logging.info("✅ Unit 7: Complete.")
    except Exception as e:
        logging.error(f"🚨 Unit 7 FAILED: {e}", exc_info=True)
        state["embedding_status"] = "FAILED"
        update_document_status(state["document_id"], "EMBEDDING_FAILED")
    return state
    
def unit_8_update_database(state: Dict[str, Any]) -> Dict[str, Any]:
    """Persists all extracted data to the database."""
    logging.info("▶️ Unit 8: Update Database")
    document_id = state.get("document_id")
    if not document_id:
        logging.error("🚨 Unit 8 FAILED: Cannot update database without a document_id.")
        return state
    try:
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                case_id = find_or_create_case(conn, state.get("analysis_json"))
                
                cur.execute(
                    "UPDATE documents SET case_id = %s, full_text = %s, cleaned_text = %s, analysis_json = %s, status = 'COMPLETED' WHERE id = %s;",
                    (case_id, state.get("full_text"), state.get("cleaned_text"), json.dumps(state.get("analysis_json")), document_id)
                )
                
                if state.get("chunks") and state.get("embeddings"):
                    for i, chunk_text in enumerate(state["chunks"]):
                        cur.execute(
                            "INSERT INTO chunks (document_id, chunk_text, embedding) VALUES (%s, %s, %s);",
                            (document_id, chunk_text, state["embeddings"][i])
                        )
                logging.info(f"   Database successfully updated for document ID: {document_id}.")
                logging.info("✅ Unit 8: Complete.")
    except Exception as e:
        logging.error(f"🚨 Unit 8 FAILED: {e}", exc_info=True)
        update_document_status(document_id, "DB_UPDATE_FAILED")
    return state
    
def unit_9_release_lease(state: Dict[str, Any]) -> None:
    """Deletes the lock file from GCS to release the lease."""
    logging.info("▶️ Unit 9: Release Lease")
    file_name = state["file_name"]
    try:
        blob = storage_client.bucket(LOCK_BUCKET_NAME).blob(f"{file_name}.lock")
        blob.delete()
        logging.info(f"   Lease released for '{file_name}'.")
        logging.info("✅ Unit 9: Complete.")
    except google_exceptions.NotFound:
        logging.warning("   Lock file was already gone. No action needed.")
    except Exception as e:
        logging.error(f"🚨 Unit 9 FAILED but non-critical: {e}", exc_info=True)

# ==================================================================================================
#    HELPER FUNCTIONS
# ==================================================================================================

def update_document_status(document_id: Optional[int], status: str) -> None:
    """Helper to update the status of a document record."""
    if not document_id: return
    try:
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE documents SET status = %s WHERE id = %s;", (status, document_id))
    except Exception as e:
        logging.error(f"Helper failed to update status to '{status}' for doc ID {document_id}: {e}")

def find_or_create_case(conn: psycopg.Connection, analysis: Optional[Dict[str, Any]]) -> Optional[int]:
    """Finds a case by micro_id or creates a new one, returning the case ID."""
    analysis = analysis or {}
    case_micro_id = analysis.get("case_micro_id", "UNK")
    case_number = analysis.get("case_number", "UNKNOWN")
    
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM cases WHERE case_micro_id = %s;", (case_micro_id,))
        result = cur.fetchone()
        if result:
            return result[0]
        else:
            cur.execute("INSERT INTO cases (case_micro_id, case_number) VALUES (%s, %s) RETURNING id;", (case_micro_id, case_number))
            new_id = cur.fetchone()
            return new_id[0] if new_id else None

#### GOLDEN RULE REMINDER ####
# THE DEPRECATED GOOGLE-GENERATIVEAI SDK IS NOT USED IN THIS SCRIPT.
# WE ARE BUILDING ON THE ENTERPRISE-GRADE GOOGLE-CLOUD-AIPLATFORM SDK.
##############################
