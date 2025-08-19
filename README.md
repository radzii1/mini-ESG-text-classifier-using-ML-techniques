# ESG Text Classifier (NLP)

**One-day project:** Classify short facility/operations notes into **Energy / Water / Waste / Transport** using **TF‑IDF + Linear SVM**.

## Why
Fast, understandable baseline that’s directly useful for ESG ticket triage and reporting.

## Data
`data/esg_samples.csv` — tiny seed dataset you can extend.  
Columns:
- `text`: short description
- `label`: one of `energy`, `water`, `waste`, `transport`

## Train
```bash
python -m venv .venv && . .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/train.py


## How to use
Train the model:
```bash
python src/train.py
