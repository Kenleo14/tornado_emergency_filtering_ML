# BERT training model for binary classification of tornado-related tweets

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # suppress TF warnings

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from pathlib import Path

# CONFIG
DATA_PATH = Path("output/processed_joplin_binary.csv")
MODEL_DIR = Path("bert-tornado-classifier")
MODEL_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Run preprocess_data.py first! Missing: {DATA_PATH}")

# LOAD DATA
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} pre-processed samples")

# Ensure clean_text is string and not empty
df["clean_text"] = df["clean_text"].astype(str)
df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)

# Map labels
label2id = {"Non-Informative": 0, "Informative": 1}
id2label = {v: k for k, v in label2id.items()}
df["label_id"] = df["label"].map(label2id)

# Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["clean_text"].tolist(),
    df["label_id"].tolist(),
    test_size=0.2,
    stratify=df["label_id"],
    random_state=42
)

print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

# TOKENIZER 
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# DATASET
class TweetDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

train_ds = TweetDataset(train_texts, train_labels)
val_ds = TweetDataset(val_texts, val_labels)

# MODEL 
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    label2id=label2id,
    id2label=id2label
)

# METRICS
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# TRAINING ARGS
training_args = TrainingArguments(
    output_dir=str(MODEL_DIR),
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    report_to=[],
    logging_steps=10,
    save_total_limit=2,
)

# TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

print(f"CUDA available: {torch.cuda.is_available()}")
trainer.train()

# FINAL EVAL
print("\n=== Validation Results ===")
results = trainer.evaluate()
print(results)

predictions = trainer.predict(val_ds).predictions.argmax(-1)
print("\n=== Classification Report ===")
print(classification_report(val_labels, predictions,
                            target_names=["Non-Informative", "Informative"]))

# SAVE MODEL
trainer.save_model(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))
print(f"\nModel saved to → {MODEL_DIR}")