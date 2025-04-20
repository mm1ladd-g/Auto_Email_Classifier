# ─────────────────────────── stage 1: builder ───────────────────────────
FROM python:3.11-slim-bookworm AS builder
LABEL stage=builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git libgomp1 libopenblas-base && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-lock.txt .

RUN python -m venv /venv && \
    /venv/bin/pip install --no-cache-dir --upgrade pip wheel && \
    /venv/bin/pip install --no-cache-dir -r requirements-lock.txt

# ─────────────────────────── stage 2: runtime ───────────────────────────
FROM python:3.11-slim-bookworm
LABEL maintainer="Mahdi M. <mahdi@example.com>" \
      org.opencontainers.image.source="https://github.com/mm1ladd-g/Auto_Email_Classifier"

ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

COPY --from=builder /venv /venv

WORKDIR /app
COPY . .

# ensure tokenizer & onnx artefacts are in image even if DVC‑ignored
COPY models/minilm-epoch3 models/minilm-epoch3
COPY models/minilm-int8.onnx models/minilm-int8.onnx

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s CMD \
  curl -f http://localhost:8000/healthz || exit 1

# two workers handle more QPS on a single vCPU without extra RAM
CMD ["uvicorn", "app.main:app", "--workers", "2", "--host", "0.0.0.0", "--port", "8000", "--lifespan", "on"]
