# Fake News Detection

A machine learning project to classify news articles as **Real** or **Fake** using NLP techniques.  
Built with Logistic Regression and a Bidirectional LSTM, with SHAP explainability.

---

## Dataset

| File | Rows | Description |
|------|------|-------------|
| `data/train.csv` | 400 | Labelled articles for training |
| `data/test.csv` | 100 | Unlabelled articles — generate predictions on these |
| `data/sample_submission.csv` | 100 | Expected submission format |

### Columns

**train.csv**
- `id` — unique article identifier
- `title` — news headline
- `text` — article body
- `label` — `1` = Fake, `0` = Real

**test.csv**
- `id`, `title`, `text` — same as above, no label

---

## Project Structure

```
fake-news-detector/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── src/
│   ├── preprocess.py        — text cleaning and tokenization
│   ├── evaluate.py          — metrics and model comparison
│   ├── explain.py           — SHAP explainability
│   └── models/
│       ├── lr_model.py      — Logistic Regression + TF-IDF
│       └── lstm_model.py    — Bidirectional LSTM (PyTorch)
├── notebooks/
│   └── 01_eda.py            — exploratory data analysis
├── results/
│   └── plots/               — confusion matrices, ROC curves, SHAP plots
├── tests/
│   └── test_preprocess.py
├── .github/workflows/
│   └── tests.yml            — CI pipeline
└── requirements.txt
```

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
```

Train the baseline model:

```bash
python src/models/lr_model.py
```

Generate predictions on test set:

```bash
python src/models/lr_model.py --predict
```

This saves `submission.csv` in the root directory.

---

## Models

### Logistic Regression + TF-IDF
- TF-IDF vectorizer with 50,000 features and bigrams
- Fast, interpretable baseline
- Validation F1: **0.94**

### Bidirectional LSTM
- 128-dim trainable embeddings
- 2-layer BiLSTM with dropout
- Captures sequential context in article text
- Validation F1: **0.97**

---

## Explainability

SHAP is used to explain individual predictions — showing which words pushed the model toward Fake or Real.

```
results/plots/lr_shap_summary.png       — global feature importance
results/plots/lr_shap_waterfall_fake.png — single article breakdown
```

---

## Submission Format

Your `submission.csv` should look like:

```
id,label
0,1
1,0
2,1
...
```

---

## Evaluation Metric

Submissions are scored on **Macro F1 Score**.

---

## Requirements

```
pandas
scikit-learn
torch
shap
matplotlib
seaborn
nltk
```

Full list in `requirements.txt`.
