from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, status
from app.loader import predict
from app.schemas import EmailRequest, EmailResponse

app = FastAPI(title="Auto Email Classifier", version="1.1.0")

# ─── tunables (can be overridden via env) ────────────────────────────────
MAX_BYTES = int(os.getenv("AEC_MAX_EMAIL_BYTES", 4_096))
CONF_THRESHOLD = float(os.getenv("AEC_CONFIDENCE_THRESHOLD", "0.50"))


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=EmailResponse)
def classify(req: EmailRequest):
    if len(req.email.encode()) > MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Email exceeds {MAX_BYTES} bytes",
        )

    category, probs = predict(req.email)

    if max(probs.values()) < CONF_THRESHOLD:
        category = "unknown"

    return {"category": category, "probabilities": probs}
