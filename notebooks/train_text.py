# notebooks/train_text.py
"""
Fine-tune RoBERTa on LIAR dataset for fake news detection.
Run from project root:  python notebooks/train_text.py
Saves model to models/roberta_fakenews/
"""

import torch
from datasets import load_dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from pathlib import Path

SAVE_PATH = "models/roberta_fakenews"
MODEL_NAME = "roberta-base"
MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5

# ─── LIAR label mapping ──────────────────────────────────────────────────────
# LIAR has 6 labels: pants-fire, false, barely-true, half-true, mostly-true, true
# We map to binary: 0=real (mostly-true, true) / 1=fake (rest)
LIAR_FAKE_LABELS = {0, 1, 2}   # pants-fire=0, false=1, barely-true=2
LIAR_REAL_LABELS = {3, 4, 5}   # half-true=3, mostly-true=4, true=5

def load_and_prepare():
    print("Loading LIAR dataset...")
    dataset = load_dataset("UKPLab/liar")

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(examples):
        texts = [str(t) for t in examples["text"]]
        tokenized = tokenizer(
            texts,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
        )
        tokenized["labels"] = [int(lbl) for lbl in examples["labels"]]
        return tokenized

    tokenized_ds = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    return tokenized_ds, tokenizer

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def train():
    dataset, tokenizer = load_and_prepare()

    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    collator = DataCollatorWithPadding(tokenizer)

    args = TrainingArguments(
        output_dir="models/roberta_checkpoints",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        report_to="none",
    )

    trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

    print("Starting training...")
    trainer.train()

    print(f"Evaluating on test set...")
    results = trainer.evaluate(dataset["test"])
    print(f"Test results: {results}")

    Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
    trainer.save_model(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")


if __name__ == "__main__":
    train()