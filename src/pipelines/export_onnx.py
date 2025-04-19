# src/pipelines/export_onnx.py
"""
Export the fine‑tuned model at models/minilm-epoch3 → 8‑bit ONNX.

Run:
    python src/pipelines/export_onnx.py
"""

from pathlib import Path
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

HF_DIR = Path("models/minilm-epoch3")
ONNX_OUT = Path("models/minilm.onnx")


def main():
    print("🔄  Loading HF model from", HF_DIR)
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        HF_DIR, export=True, file_name=ONNX_OUT.name, optimize=True
    )
    # save ONNX
    ort_model.save_pretrained(ONNX_OUT.parent)
    # copy tokenizer
    tok = AutoTokenizer.from_pretrained(HF_DIR)
    tok.save_pretrained(ONNX_OUT.parent)

    print("✔️  Exported ONNX model to", ONNX_OUT)


if __name__ == "__main__":
    main()
