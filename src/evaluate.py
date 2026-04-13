"""
Compare LR vs BiLSTM on the validation set.

Run: python src/evaluate.py
"""

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocess import get_splits

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"


def compare():
    X_train, X_val, y_train, y_val = get_splits()
    results = {}

    # LR
    lr_path = RESULTS_DIR / "lr_pipeline.pkl"
    if lr_path.exists():
        with open(lr_path, "rb") as f:
            pipe = pickle.load(f)
        proba = pipe.predict_proba(X_val)[:, 1]
        pred  = (proba >= 0.5).astype(int)
        results["Logistic Regression"] = {"proba": proba, "pred": pred}
        print("── Logistic Regression ──────────────────────")
        print(classification_report(y_val, pred, target_names=["Real", "Fake"]))

    # LSTM
    lstm_path = RESULTS_DIR / "lstm_checkpoint.pt"
    if lstm_path.exists():
        import torch
        from src.models.lstm_model import BiLSTM, TestDataset
        from torch.utils.data import DataLoader

        ckpt  = torch.load(lstm_path, map_location="cpu")
        vocab = ckpt["vocab"]
        model = BiLSTM(len(vocab))
        model.load_state_dict(ckpt["state"])
        model.eval()

        dl = DataLoader(TestDataset(X_val, vocab), batch_size=64)
        all_proba = []
        with torch.no_grad():
            for x in dl:
                all_proba.extend(torch.sigmoid(model(x)).numpy())
        proba = np.array(all_proba)
        pred  = (proba >= 0.5).astype(int)
        results["BiLSTM"] = {"proba": proba, "pred": pred}
        print("\n── BiLSTM ───────────────────────────────────")
        print(classification_report(y_val, pred, target_names=["Real", "Fake"]))

    if len(results) < 2:
        print("Train both models first.")
        return

    # ROC comparison
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, res in results.items():
        auc = roc_auc_score(y_val, res["proba"])
        RocCurveDisplay.from_predictions(y_val, res["proba"], ax=ax, name=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title("Model Comparison — ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "comparison_roc.png", dpi=150)
    plt.close()

    print("\n┌─────────────────────┬──────────┬──────────┐")
    print("│ Model               │ ROC-AUC  │ F1 Macro │")
    print("├─────────────────────┼──────────┼──────────┤")
    for name, res in results.items():
        auc = roc_auc_score(y_val, res["proba"])
        f1  = f1_score(y_val, res["pred"], average="macro")
        print(f"│ {name:<19} │ {auc:.4f}   │ {f1:.4f}   │")
    print("└─────────────────────┴──────────┴──────────┘")
    print(f"\nPlot saved → {PLOTS_DIR / 'comparison_roc.png'}")


if __name__ == "__main__":
    compare()
