# src/data/build_dataset.py
"""
Builds train/val/test CSVs from raw Enron + spam corpora using weak keyword rules.
Run:  python src/data/build_dataset.py
"""

import re, random
from pathlib import Path
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split
from langdetect import detect
from tqdm import tqdm

import label_rules as LR

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEYWORDS = {
    "support": LR.SUPPORT,
    "sales": LR.SALES,
    "partnership": LR.PARTNERSHIP,
    "spam": LR.SPAM,
}


def clean(text: str) -> str:
    """Lower‑case, strip newlines & extra spaces."""
    return re.sub(r"\s+", " ", text).strip().lower()


def assign_label(text: str) -> str | None:
    t = clean(text)
    for label, words in KEYWORDS.items():
        if any(w in t for w in words):
            return label
    return None


def load_enron() -> list[dict]:
    csv_path = next((RAW_DIR / "enron").rglob("*.csv"))
    df = pd.read_csv(csv_path)
    # find the first column that looks like message text
    text_col = next(
        (c for c in df.columns if c.lower() in {"message", "text", "body"}), "message"
    )
    rows = []
    for txt in tqdm(df[text_col].astype(str), desc="Enron CSV"):
        lbl = assign_label(txt)
        if lbl:
            try:
                lang = detect(txt[:200])
            except Exception:
                lang = "unknown"
            rows.append({"text": txt, "label": lbl, "lang": lang})
    return rows


def load_spam() -> list[dict]:
    rows = []
    # many Kaggle spam sets include a CSV called spam_emails.csv
    csv_files = list((RAW_DIR / "spam").rglob("*.csv"))
    for f in csv_files:
        df = pd.read_csv(f, encoding="utf‑8", on_bad_lines="skip")
        text_col = next(
            (c for c in df.columns if "text" in c.lower() or "body" in c.lower()), None
        )
        if text_col is None:
            continue
        for txt in df[text_col].astype(str):
            lbl = assign_label(txt)
            if lbl:
                try:
                    lang = detect(txt[:200])
                except Exception:
                    lang = "unknown"
                rows.append({"text": txt, "label": lbl, "lang": lang})
    return rows


def main():
    rows = load_enron() + load_spam()
    random.shuffle(rows)

    print("After weak labelling:", Counter(r["label"] for r in rows))

    df = pd.DataFrame(rows)
    df["id"] = range(len(df))
    X, y = df["text"], df["label"]

    train_text, temp_text, train_y, temp_y = train_test_split(
        X, y, stratify=y, test_size=0.30, random_state=42
    )
    val_text, test_text, val_y, test_y = train_test_split(
        temp_text, temp_y, stratify=temp_y, test_size=0.50, random_state=42
    )

    def to_csv(texts, labels, name):
        out = pd.DataFrame({"text": texts, "label": labels})
        out.to_csv(OUT_DIR / f"{name}.csv", index=False)

    to_csv(train_text, train_y, "train")
    to_csv(val_text, val_y, "val")
    to_csv(test_text, test_y, "test")

    print("Saved CSVs in", OUT_DIR.resolve())
    print("Train distribution:", dict(Counter(train_y)))
    print("Val   distribution:", dict(Counter(val_y)))
    print("Test  distribution:", dict(Counter(test_y)))


if __name__ == "__main__":
    main()
