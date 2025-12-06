#Kenneth Fulton 
#Dr. Liang Gongbo 
#CSCI 5341-001 

# Term project: BERT training model for binary classification of tornado-related tweets

import os
import warnings
import logging

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings("ignore")

import pandas as pd
import torch
import numpy as np
import emoji
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    logging as hf_logging
)
from torch.utils.data import Dataset

# Hide loading warnings
hf_logging.set_verbosity_error()

# CONFIGURATION 
MODEL_NAME = "vinai/bertweet-base"
# NEW: centralized name for the saved model
OUTPUT_DIR = "bertweet-joplin-informative-v1" 

# LOAD DATA
df = pd.read_csv("output/processed_joplin_binary.csv")
df = df[df["clean_text"].notna() & (df["clean_text"].str.strip() != "")].reset_index(drop=True)

# METADATA INJECTION
df["text"] = df["clean_text"].astype(str).apply(emoji.demojize)

meta_parts = []
# 1. URL signal
meta_parts.append(df["url_count"].apply(lambda x: " [HAS_URL]" if x > 0 else ""))
# 2. Mention count
meta_parts.append(df["mention_count"].apply(lambda x: f" [MENTIONS_{x}]" if x > 0 else ""))
# 3. Hashtag count
meta_parts.append(df["hashtag_count"].apply(lambda x: f" [HASHTAGS_{x}]" if x > 0 else ""))
# 4. Textual presence indicators
if "mentions_text" in df.columns:
    meta_parts.append(df["mentions_text"].fillna("").str.strip().apply(
        lambda x: " [HAS_MENTION]" if len(x) > 0 else ""
    ))
if "hashtags_text" in df.columns:
    meta_parts.append(df["hashtags_text"].fillna("").str.strip().apply(
        lambda x: " [HAS_HASHTAG]" if len(x) > 0 else ""
    ))

df["text"] = df["text"] + " " + pd.concat(meta_parts, axis=1).apply(lambda row: "".join(row), axis=1)
df["text"] = df["text"].str.replace(r'\s+', ' ', regex=True).str.strip()

# Labels & Class Weights
label2id = {"Non-Informative": 0, "Informative": 1}
df["label_id"] = df["label"].map(label2id)

class_weights = torch.tensor(
    compute_class_weight("balanced", classes=np.array([0, 1]), y=df["label_id"]),
    dtype=torch.float
).to("cuda" if torch.cuda.is_available() else "cpu")

# Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"], df["label_id"], test_size=0.2, stratify=df["label_id"], random_state=42
)

# DATASET & MODEL
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, normalization=True)

class TweetDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(list(texts), truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        self.labels = torch.tensor(labels.values, dtype=torch.long)
    
    def __len__(self): return len(self.labels)
    
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.encodings.items()}
        item["labels"] = self.labels[i]
        return item

train_ds = TweetDataset(train_texts, train_labels)
val_ds = TweetDataset(val_texts, val_labels)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, label2id=label2id, id2label={v:k for k,v in label2id.items()}
)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.CrossEntropyLoss(weight=class_weights)(
            outputs.logits.view(-1, 2), labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,  # Uses variable
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=200,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    label_smoothing_factor=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_strategy="no",
    report_to=[],
    fp16=torch.cuda.is_available(),
    seed=42,
)

trainer = WeightedTrainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds)

# TRAIN
if __name__ == "__main__":
    print(f"Training model: {OUTPUT_DIR}")
    print(f"Data size: {len(train_ds)} samples")
    
    trainer.train()

    print("\n" + "="*80)
    print(f"FINAL RESULT: {OUTPUT_DIR}")
    print("="*80)
    preds = trainer.predict(val_ds).predictions.argmax(axis=1)
    print(classification_report(val_labels, preds, target_names=["Non-Informative", "Informative"], digits=4))

    # Save using the variable to ensure consistency
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel and tokenizer saved to ./{OUTPUT_DIR}")
