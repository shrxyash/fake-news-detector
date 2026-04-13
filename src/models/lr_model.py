"""
Logistic Regression + TF-IDF baseline.

Train:   python src/models/lr_model.py
Predict: python src/models/lr_model.py --predict
"""

import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.preprocess import get_splits, load_test

ROOT        = Path(__file__).parent.parent.parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"
MODEL_PATH  = RESULTS_DIR / "lr_pipeline.pkl"


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])


def train():
    X_train, X_val, y_train, y_val = get_splits()

    pipeline = build_pipeline()
    print("Training ...")
    pipeline.fit(X_train, y_train)

    val_pred  = pipeline.predict(X_val)
    val_proba = pipeline.predict_proba(X_val)[:, 1]

    print("\n── Validation ─────────────────────────────────")
    print(classification_report(y_val, val_pred, target_names=["Real", "Fake"]))
    print(f"ROC-AUC: {roc_auc_score(y_val, val_proba):.4f}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_val, val_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Logistic Regression — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "lr_confusion_matrix.png", dpi=150)
    plt.close()

    # ROC curve
    auc = roc_auc_score(y_val, val_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_val, val_proba, ax=ax, name=f"LR (AUC={auc:.3f})")
    ax.set_title("Logistic Regression — ROC Curve")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "lr_roc_curve.png", dpi=150)
    plt.close()

    print(f"Plots saved to {PLOTS_DIR}")

    # Save model
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved → {MODEL_PATH}")

    return pipeline


def predict_test():
    if not MODEL_PATH.exists():
        print("Model not found. Run training first: python src/models/lr_model.py")
        sys.exit(1)

    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)

    df = load_test()
    proba = pipeline.predict_proba(df["clean"].values)[:, 1]
    preds = (proba >= 0.5).astype(int)

    submission = pd.DataFrame({"id": df["id"], "label": preds})
    out = ROOT / "submission.csv"
    submission.to_csv(out, index=False)
    print(f"submission.csv saved → {out}")
    print(submission["label"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", action="store_true", help="Generate test predictions")
    args = parser.parse_args()

    if args.predict:
        predict_test()
    else:
        train()
