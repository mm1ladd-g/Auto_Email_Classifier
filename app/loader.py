"""Model loader & predictor (cached).

* Loads the quantised INT‑8 ONNX model once per process.
* Uses numerically‑stable soft‑max (scipy).
* Lets you override artefact locations with environment variables.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import onnxruntime as ort
from scipy.special import softmax
from transformers import AutoTokenizer

LABELS: List[str] = ["support", "sales", "partnership", "spam"]

# ─── artefact locations ──────────────────────────────────────────────────
ONNX_PATH = Path(
    os.getenv("ONNX_MODEL_PATH", "models/minilm-int8.onnx")  # <- keep in sync with export script
)
TOKENIZER_PATH = Path(os.getenv("TOKENIZER_DIR", "models/minilm-epoch3"))

# ─── cached loaders ──────────────────────────────────────────────────────
@cache
def _load() -> Tuple[ort.InferenceSession, AutoTokenizer]:
    sess = ort.InferenceSession(
        ONNX_PATH.as_posix(),
        providers=["CPUExecutionProvider"],
    )
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    # free trainer object (saves ≈ 30 MB RSS)
    if hasattr(tok.backend_tokenizer, "model") and hasattr(
        tok.backend_tokenizer.model, "trainer"
    ):
        tok.backend_tokenizer.model.trainer = None
    return sess, tok


# ─── public API ──────────────────────────────────────────────────────────
def predict(text: str) -> Tuple[str, Dict[str, float]]:
    """Return (best_label, {label:prob, …}) for a single e‑mail body."""
    sess, tok = _load()
    inputs = tok(text, return_tensors="np", truncation=True, max_length=256)
    logits: np.ndarray = sess.run(None, dict(inputs))[0][0]
    probs = softmax(logits).astype(float)
    idx = int(np.argmax(probs))
    return LABELS[idx], {lbl: float(p) for lbl, p in zip(LABELS, probs)}
