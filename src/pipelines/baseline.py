#/src/pipelines/baseline.py

"""
Baseline model: TF‑IDF  ‑>  Multinomial Naïve Bayes
Usage:  python src/pipelines/baseline.py
Outputs:
    models/baseline.joblib           (sklearn pipeline)
    reports/baseline_metrics.json    (accuracy + macro‑F1)
"""

import json
import pathlib
import joblib
import sys
from collections import Counter

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, f1_score

DATA_DIR = pathlib.Path("data/processed")
MODEL_DIR = pathlib.Path("models")
REPORT_DIR = pathlib.Path("reports")
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def load(split: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(DATA_DIR / f"{split}.csv")
    return df["text"], df["label"]


def main():
    X_train, y_train = load("train")
    X_val, y_val = load("val")  # we’ll treat val as test for baseline

    print("Train class counts:", Counter(y_train))
    pipe = make_pipeline(
        TfidfVectorizer(stop_words="english", max_features=50_000, ngram_range=(1, 2)),
        MultinomialNB(),
    )
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_val)
    acc = accuracy_score(y_val, preds)
    macro_f1 = f1_score(y_val, preds, average="macro")

    print(f"\nValidation accuracy: {acc:.4f}")
    print(f"Validation macro‑F1: {macro_f1:.4f}\n")
    print(classification_report(y_val, preds, digits=3))

    # save artefacts
    joblib.dump(pipe, MODEL_DIR / "baseline.joblib")
    json.dump(
        {"accuracy": acc, "macro_f1": macro_f1},
        open(REPORT_DIR / "baseline_metrics.json", "w"),
        indent=2,
    )
    print("\n✔️  Saved model to", MODEL_DIR / "baseline.joblib")
    print("✔️  Saved metrics to", REPORT_DIR / "baseline_metrics.json")


if __name__ == "__main__":
    sys.exit(main())
