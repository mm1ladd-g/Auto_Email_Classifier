# app/main.py
from fastapi import FastAPI
from app.loader import predict
from app.schemas import EmailRequest, EmailResponse

app = FastAPI(title="Auto Email Classifier", version="1.0.0")


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=EmailResponse)
def classify(req: EmailRequest):
    category, probs = predict(req.email)
    return {"category": category, "probabilities": probs}
