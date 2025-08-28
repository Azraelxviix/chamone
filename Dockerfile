#### GOLDEN RULE FOR THIS SCRIPT ####
# WE ARE NOT USING THE GOOGLE-GENERATIVEAI SDK.
# ALL GEMINI INTERACTIONS ARE HANDLED THROUGH THE ROBUST GOOGLE-CLOUD-AIPLATFORM SDK.
#####################################

# --- STAGE 1: The 'builder' Stage ---
# This stage installs dependencies, including build-time tools.
FROM python:3.12-slim-bookworm AS builder

WORKDIR /app

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies required for building Python packages (like psycopg)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- STAGE 2: The 'runtime' Stage ---
# This stage creates the final, lean image for production.
FROM python:3.12-slim-bookworm

WORKDIR /app

# Install only the runtime system dependencies (libpq5 is needed by psycopg)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 && \
    rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application code
COPY main.py .

# Set the command to run the application using functions-framework
# This is the entry point for Cloud Run when triggered by an event.
CMD ["functions-framework", "--target=on_cloud_event", "--port=8080"]

#### GOLDEN RULE REMINDER ####
# THE DEPRECATED GOOGLE-GENERATIVEAI SDK IS NOT USED IN THIS BUILD.
# WE ARE BUILDING ON THE ENTERPRISE-GRADE GOOGLE-CLOUD-AIPLATFORM SDK.
##############################
