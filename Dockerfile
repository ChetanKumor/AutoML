# syntax=docker/dockerfile:1
#
# Multi-stage build: dependencies are compiled into a wheel cache in the first
# stage so the runtime image carries neither build toolchains nor pip caches.

FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# LightGBM needs libgomp to build and to run.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt


FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    AUTOML_MODEL_DIR=/app/saved_models \
    AUTOML_LOG_DIR=/app/logs

# libgomp is a runtime requirement of LightGBM; curl backs the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Run as a non-root user.
RUN useradd --create-home --uid 1000 automl
COPY --chown=automl:automl . .
RUN mkdir -p /app/saved_models /app/saved_encoders /app/logs \
    && chown -R automl:automl /app
USER automl

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
