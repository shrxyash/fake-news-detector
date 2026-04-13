"""
SHAP explainability for the Logistic Regression model.

Run: python src/explain.py
"""

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocess import get_splits

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"
MODEL_PATH  = RESULTS_DIR / "lr_pipeline.pkl"


def explain_lr():
    if not MODEL_PATH.exists():
        print("Model not found. Train first: python src/models/lr_model.py")
        return

    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)

    X_train, X_val, y_train, y_val = get_splits()
    tfidf         = pipeline["tfidf"]
    clf           = pipeline["clf"]
    feature_names = tfidf.get_feature_names_out()

    try:
        import shap
        X_sample = tfidf.transform(X_val[:200])
        explainer   = shap.LinearExplainer(clf, X_sample, feature_perturbation="interventional")
        shap_values = explainer(X_sample)

        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        # Summary plot
        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values.values, X_sample, feature_names=feature_names,
                          max_display=20, show=False)
        plt.title("SHAP — Top 20 features (Logistic Regression)")
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "lr_shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved → {PLOTS_DIR / 'lr_shap_summary.png'}")

        # Waterfall for one fake and one real
        for target, name in [(1, "fake"), (0, "real")]:
            idx = np.where(y_val[:200] == target)[0]
            if len(idx):
                fig = plt.figure(figsize=(10, 5))
                shap.plots.waterfall(shap_values[idx[0]], max_display=15, show=False)
                plt.title(f"SHAP Waterfall — {name.upper()} article")
                plt.tight_layout()
                fig.savefig(PLOTS_DIR / f"lr_shap_waterfall_{name}.png", dpi=150, bbox_inches="tight")
                plt.close()
                print(f"Saved → {PLOTS_DIR / f'lr_shap_waterfall_{name}.png'}")

    except ImportError:
        # Fallback: coefficient bar chart
        print("SHAP not installed — generating coefficient importance plot instead.")
        coef = clf.coef_[0]
        top_fake = np.argsort(coef)[-20:][::-1]
        top_real = np.argsort(coef)[:20]

        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, indices, title, color in [
            (axes[0], top_fake, "Words → FAKE", "#e07070"),
            (axes[1], top_real, "Words → REAL", "#70a0d0"),
        ]:
            words  = [feature_names[i] for i in indices]
            values = np.abs(coef[indices])
            ax.barh(words[::-1], values[::-1], color=color)
            ax.set_title(title); ax.set_xlabel("|coefficient|")
        fig.suptitle("Logistic Regression — Feature Importance")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "lr_feature_importance.png", dpi=150)
        plt.close()
        print(f"Saved → {PLOTS_DIR / 'lr_feature_importance.png'}")


if __name__ == "__main__":
    explain_lr()
