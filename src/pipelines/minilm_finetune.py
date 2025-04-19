# src/pipelines/minilm_finetune.py
"""
Fine‑tunes sentence‑transformers/all-MiniLM‑L6‑v2 on our 4‑class dataset.
CPU‑only: 3 epochs ≈ 20 min on an M‑series Mac.

Run:
    python src/pipelines/minilm_finetune.py
"""

import json
import os
import pathlib
import torch
from collections import Counter

from datasets import Dataset
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

# ─── constants ──────────────────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_DIR = pathlib.Path("data/processed")
MODEL_DIR = pathlib.Path("models/minilm-epoch3")
REPORT_DIR = pathlib.Path("reports")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LABEL2ID = {"support": 0, "sales": 1, "partnership": 2, "spam": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ─── data helpers ───────────────────────────────────────────────────────
def load_split(split: str) -> Dataset:
    df = pd.read_csv(DATA_DIR / f"{split}.csv").sample(frac=1, random_state=42)
    ds = Dataset.from_pandas(
        df[["text", "label"]].rename(columns={"text": "sentence"}),
        preserve_index=False,  # <-- key fix: drop index column
    )
    return ds


def preprocess(ds: Dataset, tok) -> Dataset:
    ds = ds.map(
        lambda ex: tok(ex["sentence"], truncation=True),
        batched=True,
        remove_columns=["sentence"],
    )
    return ds.map(
        lambda ex: {"labels": LABEL2ID[ex["label"]]},
        remove_columns=["label"],
    )


# ─── Lightning module ───────────────────────────────────────────────────
class LitMiniLM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            num_labels=4,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        self.val_acc = MulticlassAccuracy(num_classes=4)
        self.val_f1 = MulticlassF1Score(num_classes=4, average="macro")

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, _):
        out = self(**batch)
        self.log("train_loss", out.loss, on_step=True, prog_bar=True)
        return out.loss

    def validation_step(self, batch, _):
        out = self(**batch)
        preds = out.logits.argmax(-1)
        self.val_acc.update(preds, batch["labels"])
        self.val_f1.update(preds, batch["labels"])

    def on_validation_epoch_end(self):
        self.log("val_accuracy", self.val_acc.compute(), prog_bar=True)
        self.log("val_macro_f1", self.val_f1.compute(), prog_bar=True)
        self.val_acc.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-5)


def main():
    lit = LitMiniLM()
    tok = lit.tokenizer
    train_ds = preprocess(load_split("train"), tok)
    val_ds = preprocess(load_split("val"), tok)

    print("Class counts (train):", Counter(train_ds["labels"]))

    collate = DataCollatorWithPadding(tok, return_tensors="pt")
    dl_train = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate)
    dl_val = DataLoader(val_ds, batch_size=8, collate_fn=collate)

    trainer = pl.Trainer(accelerator="cpu", max_epochs=3, log_every_n_steps=50)
    trainer.fit(lit, dl_train, dl_val)

    lit.model.save_pretrained(MODEL_DIR)
    tok.save_pretrained(MODEL_DIR)

    metrics = {
        "val_accuracy": trainer.callback_metrics["val_accuracy"].item(),
        "val_macro_f1": trainer.callback_metrics["val_macro_f1"].item(),
    }
    print("Validation metrics:", metrics)
    json.dump(metrics, open(REPORT_DIR / "minilm_metrics.json", "w"), indent=2)
    print("✔️  Saved HF model dir to", MODEL_DIR)
    print("✔️  Saved metrics JSON to", REPORT_DIR / "minilm_metrics.json")


if __name__ == "__main__":
    main()
