# Data Preprocessing Script for Joplin 2011 Tweets

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from pathlib import Path
from collections import Counter
import string

# CONFIG
BASE = Path(r"ISCRAM2013_dataset/ISCRAM2013_dataset/Joplin_2011_labeled_data")

FILES = {
    "personal": "01_personal-informative-other/a131709.csv",
    "general":  "02_informative_caution-infosrc-donation-damage-other/a121571.csv",
    "caution":  "03_caution-n-advice_classify-extract/a122047.csv",
    "damage":   "03_damage-n-casualties_classify-extract/a126730.csv",
    "donation": "03_donation-help_classify-extract/a126728.csv",
    "infosrc":  "03_infosrc_classify-extract/a122582.csv",
}

OUT = Path("output")
OUT.mkdir(exist_ok=True)
CSV_OUT = OUT / "processed_joplin_binary.csv"
TOP_WORDS_OUT = OUT / "top_words.txt"
CONF_TH = 0.8

# LOAD AND STANDARDIZE EACH FILE
standard_dfs = []
for src, rel in FILES.items():
    path = BASE / rel
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    df = pd.read_csv(path, low_memory=False)
    print(f"{src:12} → {len(df):4} rows")

    txt = next((c for c in ["text", "text_no_rt", "tweet"] if c in df.columns), None)
    if not txt:
        print(f"  No text column in {src} – skipping")
        continue

    conf = next((c for c in df.columns if "confidence" in c.lower()), None)
    label_cols = [c for c in df.columns if any(k in c.lower() for k in ["choose_one", "predicted", "category", "type_of", "people_or"])]
    label_col = label_cols[0] if label_cols else None
    if not label_col:
        print(f"  No label column in {src} – skipping")
        continue

    std = pd.DataFrame()
    std["text"] = df[txt]
    std["confidence"] = df[conf].astype(float) if conf else 1.0
    std["raw_label"] = df[label_col]
    standard_dfs.append(std)

# CONCATENATE
if not standard_dfs:
    raise ValueError("No valid data found in any file")

raw = pd.concat(standard_dfs, ignore_index=True)
print(f"Concatenated → {len(raw)} rows")

# BINARY MAPPING
task1_informative = {
    "Informative (Direct or Indirect)",
    "Informative (Direct)",
    "Informative (Indirect)",
    "Informative (Direct or Indirect)"
}
task2_informative = {
    "Caution and advice",
    "Casualties and damage",
    "Donations of money, goods or services",
    "People missing, found, or seen",
    "Information source"
}

def to_binary(label):
    s = str(label).strip()
    return "Informative" if s in task1_informative or s in task2_informative else "Non-Informative"

raw["label"] = raw["raw_label"].apply(to_binary)

# FILTER + CLEAN
df = raw[raw["confidence"] >= CONF_TH].copy()
df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
print(f"After confidence ≥ {CONF_TH}: {len(df)} rows")

def clean_text(t):
    if not isinstance(t, str):
        return ""
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"#\w+", "", t)
    t = re.sub(r"[{}]".format(string.punctuation), "", t)  # Remove punctuation
    t = re.sub(r"\s+", " ", t).strip().lower()  # Lowercase
    return t

df["clean_text"] = df["text"].astype(str).apply(clean_text)
df = df[df["clean_text"].str.strip() != ""]
df = df.dropna(subset=["clean_text", "label"]).reset_index(drop=True)
df = df[["clean_text", "label"]].copy()

print(" ")
print(f"Final clean samples: {len(df)}")

# SAVE 
df.to_csv(CSV_OUT, index=False)
print(f"Processed data saved → {CSV_OUT}")

# TOP 20 WORDS PER LABEL
def get_top_words(texts, n=20):
    all_words = []
    for text in texts:
        all_words.extend(text.split())
    counter = Counter(all_words)
    return counter.most_common(n)

print(" ")
print("MOST COMMON WORDS PER LABEL:")

with open(TOP_WORDS_OUT, "w", encoding="utf-8") as f:
    for label in ["Informative", "Non-Informative"]:
        texts = df[df["label"] == label]["clean_text"]
        top_words = get_top_words(texts, 20)
        
        print(f"\n---  {label.upper()} ---")
        f.write(f"\n--- {label.upper()} ---\n")
        
        for word, count in top_words:
            line = f"{word:<15} {count:>5}"
            print(line)
            f.write(line + "\n")

print(f"\nTop words saved → {TOP_WORDS_OUT}")
print("="*60)

# EDA PLOTS
def save_plot(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)

# 1. Class distribution
fig, ax = plt.subplots(figsize=(7, 4))
sns.countplot(data=df, x="label", ax=ax, hue="label", palette="viridis")
ax.set_title(f"Label Distribution (Total={len(df)})")
ax.set_xlabel("Label")
ax.set_ylabel("Count")
save_plot(fig, "class_distribution.png")

# 2. Length distribution
df["length"] = df["clean_text"].str.split().str.len()
fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(
    data=df,
    x="length",
    hue="label",
    bins=40,
    kde=True,
    alpha=0.7,
    multiple="stack",
    palette="viridis",
    ax=ax,
    legend=True
)
ax.set_xlim(0, 60)
ax.set_title("Tweet Length by Label")
ax.set_xlabel("Number of Words")
ax.set_ylabel("Count")
save_plot(fig, "length_distribution.png")

print(f"\nEDA plots saved in → {OUT}")
print("PREPROCESSING COMPLETE")