# tornado_emergency_filtering_ML
Sub-Topic/Title: Deep Learning for Social Media Tornado Emergency Filtering 
Dataset: Crisis NLP https://crisisnlp.qcri.org/data/iscram2013_nuggets_datasset/ISCRAM2013_dataset.zip Description: Human-labeled tweets collected during the 2011 Joplin Missouri tornado

# Tornado Emergency Tweet Filtering (BERT)

Deep learning pipeline to classify social media posts (tweets) as Informative vs. Non-Informative for tornado-related emergencies.

- Language: Python
- Model: BERT (bert-base-uncased)
- Task: Binary text classification
- Dataset: CrisisNLP ISCRAM2013 (2012 Hurricane Sandy, 2011 Joplin tornado)

---

## Overview

This project trains a BERT-based classifier to filter crisis-related tweets and identify informative content during tornado events. It includes:

- Preprocessing script to clean and label data
- Training script using Hugging Face Transformers
- Metrics reporting (Accuracy, F1)
- Saved model artifacts for reuse/inference

---

## Repository Structure

- `preprocess_data.py` — cleans and prepares the raw dataset for training
- `train_model.py` — trains and evaluates a BERT classifier
- `ISCRAM2013_dataset/` — directory for raw dataset files (place the downloaded data here)
- `output/` — created by preprocessing; contains `processed_joplin_binary.csv` used by training
- `bert-tornado-classifier/` — created after training; contains the saved model and tokenizer

---

## Dataset

- Source: CrisisNLP ISCRAM2013 Nuggets Dataset  
  https://crisisnlp.qcri.org/data/iscram2013_nuggets_datasset/ISCRAM2013_dataset.zip

Description: Human-labeled tweets collected during the 2011 Joplin tornado.

Please adhere to the dataset’s terms of use and citation requirements.

---

## Quickstart

### 1) Environment setup

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install torch transformers scikit-learn pandas
```

GPU is optional. If CUDA is available, mixed precision (fp16) is enabled automatically.

### 2) Download the dataset

Download and extract the ISCRAM2013 dataset into `ISCRAM2013_dataset/`.

Your directory should look like:

```
ISCRAM2013_dataset/
  <raw files from the zip>
```

### 3) Preprocess

```bash
python preprocess_data.py
```

This should create:

```
output/processed_joplin_binary.csv
```

The training script reads from this path by default.

### 4) Train

```bash
python train_model.py
```

You should see logs including training/validation metrics per epoch, a final classification report, and saved artifacts in:

```
bert-tornado-classifier/
```

---

## Configuration

Key settings in `train_model.py`:

```python
# Data & model paths
DATA_PATH = Path("output/processed_joplin_binary.csv")
MODEL_DIR = Path("bert-tornado-classifier")

# TrainingArguments (epochs, batch size, etc.)
num_train_epochs=10
per_device_train_batch_size=16
per_device_eval_batch_size=16
eval_strategy="epoch"
save_strategy="epoch"
metric_for_best_model="f1"
fp16=torch.cuda.is_available()
```

If your processed CSV exists elsewhere (e.g., repository root), either move it to `output/` or update `DATA_PATH`. For example:

```python
DATA_PATH = Path("processed_joplin_binary.csv")
```

Expected columns in the processed CSV:
- `clean_text` — cleaned tweet text
- `label` — string label: `"Informative"` or `"Non-Informative"`

Label mapping (in training):
- Non-Informative → 0
- Informative → 1

---

## Troubleshooting

- Missing data error:
  - If you see: `Run preprocess_data.py first! Missing: output/processed_joplin_binary.csv`
  - Ensure preprocessing was run and the file exists at `output/processed_joplin_binary.csv`, or update `DATA_PATH` accordingly.

- Package versions:
  - Ensure `torch`, `transformers`, `scikit-learn`, and `pandas` are installed and up to date.

- GPU/Memory:
  - Reduce `max_length` (tokenizer) or batch sizes if you run out of memory.
  - CPU training is supported but slower; CUDA enables `fp16` automatically.

---

## Team

- Kenneth Fulton

## Acknowledgments

- CrisisNLP team for the ISCRAM2013 dataset.
- Hugging Face Transformers for model and training utilities.

## License

If a license file is not present, usage defaults to "all rights reserved." Add a LICENSE file to define terms for code usage and contributions.
