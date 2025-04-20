# src/pipelines/export_onnx.py
"""
Convert the fine‑tuned MiniLM to ONNX (fp32) + INT‑8.

Run:
    python src/pipelines/export_onnx.py
"""

from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

HF_DIR    = Path("models/minilm-epoch3")
OUT_DIR   = Path("models")
FP32_FILE = OUT_DIR / "model.onnx"
INT8_FILE = OUT_DIR / "minilm-int8.onnx"

def main():
    print("🔄  Exporting HF → ONNX …")
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        HF_DIR,
        export=True,          # <-- current, stable flag
    )
    ort_model.save_pretrained(OUT_DIR)    # writes model.onnx

    print("⚙️  Quantising INT‑8 …")
    quantize_dynamic(
        model_input=str(FP32_FILE),
        model_output=str(INT8_FILE),
        weight_type=QuantType.QInt8,
    )

    AutoTokenizer.from_pretrained(HF_DIR).save_pretrained(OUT_DIR)

    print("✔️  fp32  →", FP32_FILE)
    print("✔️  int8  →", INT8_FILE)

if __name__ == "__main__":
    main()
