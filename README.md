# 📧 Auto Email Classifier 🚀

[![CI](https://github.com/mm1ladd-g/Auto_Email_Classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/mm1ladd-g/Auto_Email_Classifier/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/mm1ladd-g/Auto_Email_Classifier/branch/main/graph/badge.svg)](https://codecov.io/gh/mm1ladd-g/Auto_Email_Classifier)
[![ruff](https://img.shields.io/badge/code‑style‑ruff-%2314b?logo=python&logoColor=white)](https://github.com/astral‑sh/ruff)

> **One‑click micro‑service that auto‑routes incoming e‑mails into four business‑critical buckets**
> `support | sales | partnership | spam (+ unknown for low‑confidence)`

Optimised for **reproducibility**, **MLOps hygiene** and **Apple‑Silicon performance**.
Total build time ≈ 14 h (including weak‑labelling & fine‑tune).

---

## Table of Contents

1. [Features](#features)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Dataset &amp; Modelling](#dataset--modelling)
5. [API Reference](#api-reference)
6. [Testing &amp; CI](#testing--ci)
7. [Deployment](#deployment)
8. [Security &amp; Robustness](#security--robustness)
9. [Roadmap](#roadmap)
10. [License &amp; Credits](#license--credits)

---

## Features

* **Quantised MiniLM (INT‑8 ONNX)** – 20 ms / e‑mail on 1 vCPU.
* **Weak‑labelled Enron + spam corpus** (170 k mails) with keyword precedence.
* **Confidence gating** – returns `"unknown"` below threshold.
* **Docker‑first**: multi‑arch image (arm64 & x86‑64) < 260 MB.
* **97 % test coverage**, `ruff` + `black` formatting, `mypy --strict` typing.
* **Single‑command retrain/export** via Makefile.

---

## Quick Start

### Prerequisites

* Python 3.11
* (optional) Docker ≥ 24
* (optional) DVC ≥ 3 for lazy data/model pulls

### Local dev

```bash
git clone https://github.com/mm1ladd-g/Auto_Email_Classifier.git
cd Auto_Email_Classifier

python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements-lock.txt

# pull processed data & models (≈ 180 MB), or run pipelines yourself
dvc pull

uvicorn app.main:app --reload            # http://127.0.0.1:8000/docs
```

### Docker

docker build -t email-classifier:latest .
docker run -p 8000:8000 email-classifier:latest
curl -X POST localhost:8000/predict
    -H "Content-Type: application/json"
    -d '{"email":"Need help with my invoice"}' | jq

## Configuration

All tunables are exposed via environment variables (defaults in parentheses):

Variable | Description
ONNX_MODEL_PATH (models/minilm-int8.onnx) | Path to model artefact
TOKENIZER_DIR (models/minilm-epoch3) | Hugging Face tokenizer dir
AEC_MAX_EMAIL_BYTES (4096) | Max input size – requests above are 413
AEC_CONFIDENCE_THRESHOLD (0.50) | Below → label set to unknown

## Dataset & Modelling

### Corpus overview

Source | Rows after weak‑labelling | License
Enron (Kaggle acsariyildiz/…) | 136 k | Public domain
Multiple spam CSVs |  33 k | CC BY‑NC‑ND 4.0
Synthetic (GPT‑4 few‑shot) |  2 k | MIT

### Modelling pipeline

Stage | Tooling | Val Accuracy | Val Macro‑F1
Baseline (TF‑IDF + MultinomialNB) | scikit‑learn |  0.806 |  0.468
Fine‑tuned MiniLM (3 epochs) | HF Transformers + PyTorch Lightning | 0.890 ± 0.01 | 0.73 ± 0.01

Weights exported to 8‑bit ONNX ( **60 MB** ).

Reproduce locally:

# build dataset

python src/data/build_dataset.py

# baseline

python src/pipelines/baseline.py

# fine‑tune MiniLM (Apple Silicon GPU ⇒ add --accelerator='gpu')

python src/pipelines/minilm_finetune.py

# export & quantise

python src/pipelines/export_onnx.py

API Reference

Method | Path | Req Body | Resp 200
GET | /healthz | – | { "status": "ok" }
POST | /predict | { "email": "text…" } | { "category": "support", "probabilities": { … } }

Low‑confidence outputs return** **`"unknown"` as** **`category`.

Testing & CI

make lint         # ruff, black --check, mypy --strict
make test         # pytest + coverage

GitHub Actions runs the full matrix ( **linux / macOS** , 3.11, arm64 & x86‑64) and uploads coverage to Codecov.

## Deployment


### Container

* Two Uvicorn workers (`--workers 2`) maximise QPS on a single vCPU.
* `HEALTHCHECK` hits** **`/healthz`; container starts with** **`--lifespan on` so FastAPI only reports “healthy” after the model is loaded.

### Kubernetes (optional)


kubectl apply -f docker/k8s.yaml    # includes HPA (cpu‑based)


## Security & Robustness

* **Strict Pydantic schemas** – unknown JSON keys are rejected (`400`).
* **Size guard** – requests over 4 kB →** **`413 Payload Too Large`.
* **Pinned deps** –** **`requirements-lock.txt`.
* **Readonly filesystem** – container runs as non‑root, FS mounted** **`ro` by default.
* **Confidence gating** – prevents wrong auto‑routing when the model is unsure.


## Roadmap

* [ ] Active‑learning loop with human‐in‑the‑loop UI
* [ ] `/explain` endpoint (SHAP) for per‑email transparency
* [ ] Metal f16 fine‑tune to halve latency on M‑series chips



## License & Credits

Code & synthetic data © 2025 Milad Ghavampoori – MIT License.

Enron corpus is public domain; spam CSVs under CC BY‑NC‑ND 4.0.

Big shout‑out to the OSS community: Hugging Face, FastAPI, DVC, ONNX Runtime ❤️
