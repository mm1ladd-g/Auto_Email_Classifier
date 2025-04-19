# app/loader.py
from functools import cache
from pathlib import Path
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

LABELS = ["support", "sales", "partnership", "spam"]
ONNX_PATH = Path("models/minilm.onnx")
TOKENIZER_PATH = Path("models/minilm-epoch3")  # tokenizer files


@cache
def _load():
    sess = ort.InferenceSession(
        ONNX_PATH.as_posix(), providers=["CPUExecutionProvider"]
    )
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    return sess, tok


def predict(text: str) -> tuple[str, dict]:
    sess, tok = _load()
    inputs = tok(text, return_tensors="np", truncation=True, max_length=256)
    logits = sess.run(None, dict(inputs))[0][0]
    probs = np.exp(logits) / np.exp(logits).sum()
    idx = int(np.argmax(probs))
    return LABELS[idx], {lbl: float(p) for lbl, p in zip(LABELS, probs)}
