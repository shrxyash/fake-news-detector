"""
Bidirectional LSTM classifier.

Train:   python src/models/lstm_model.py
Predict: python src/models/lstm_model.py --predict
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.preprocess import get_splits, load_test

ROOT        = Path(__file__).parent.parent.parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"
CKPT_PATH   = RESULTS_DIR / "lstm_checkpoint.pt"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Vocabulary ────────────────────────────────────────────────────────────────

class Vocabulary:
    def __init__(self, max_vocab=40_000):
        self.max_vocab = max_vocab
        self.w2i = {"<PAD>": 0, "<UNK>": 1}

    def build(self, texts):
        counter = Counter(w for t in texts for w in t.split())
        for word, _ in counter.most_common(self.max_vocab - 2):
            self.w2i[word] = len(self.w2i)
        print(f"Vocab size: {len(self.w2i):,}")
        return self

    def encode(self, text, max_len=300):
        ids = [self.w2i.get(w, 1) for w in text.split()[:max_len]]
        ids += [0] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.w2i)


# ── Dataset ───────────────────────────────────────────────────────────────────

class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=300):
        self.items = [
            (torch.tensor(vocab.encode(t, max_len), dtype=torch.long),
             torch.tensor(y, dtype=torch.float))
            for t, y in zip(texts, labels)
        ]
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


class TestDataset(Dataset):
    def __init__(self, texts, vocab, max_len=300):
        self.items = [torch.tensor(vocab.encode(t, max_len), dtype=torch.long) for t in texts]
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


# ── Model ─────────────────────────────────────────────────────────────────────

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden=256, layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.drop = nn.Dropout(0.4)
        self.fc   = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        e = self.drop(self.embedding(x))
        _, (h, _) = self.lstm(e)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(self.drop(h)).squeeze(1)


# ── Training ──────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if train: optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += ((torch.sigmoid(logits) >= 0.5).long() == y.long()).sum().item()
            total += len(y)
    return total_loss / total, correct / total


def train():
    X_train, X_val, y_train, y_val = get_splits()

    vocab = Vocabulary().build(X_train)
    train_dl = DataLoader(NewsDataset(X_train, y_train, vocab), batch_size=64, shuffle=True)
    val_dl   = DataLoader(NewsDataset(X_val,   y_val,   vocab), batch_size=64)

    model     = BiLSTM(len(vocab)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    best_acc, history = 0, []
    print(f"\nTraining on {DEVICE}")
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>9}  {'Val Acc':>8}")
    print("─" * 55)

    for epoch in range(1, 11):
        tr_loss, tr_acc = run_epoch(model, train_dl, optimizer, criterion, train=True)
        vl_loss, vl_acc = run_epoch(model, val_dl,   optimizer, criterion, train=False)
        scheduler.step(vl_loss)
        history.append((epoch, tr_loss, tr_acc, vl_loss, vl_acc))
        print(f"{epoch:>5}  {tr_loss:>10.4f}  {tr_acc:>9.4f}  {vl_loss:>9.4f}  {vl_acc:>8.4f}")

        if vl_acc > best_acc:
            best_acc = vl_acc
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({"state": model.state_dict(), "vocab": vocab}, CKPT_PATH)
            print(f"  ↑ checkpoint saved (acc {best_acc:.4f})")

    # Plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    epochs = [h[0] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [h[1] for h in history], label="Train")
    axes[0].plot(epochs, [h[3] for h in history], label="Val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(epochs, [h[2] for h in history], label="Train")
    axes[1].plot(epochs, [h[4] for h in history], label="Val")
    axes[1].set_title("Accuracy"); axes[1].legend()
    fig.suptitle("BiLSTM Training History"); fig.tight_layout()
    fig.savefig(PLOTS_DIR / "lstm_training_history.png", dpi=150); plt.close()
    print(f"Plots saved → {PLOTS_DIR}")


def predict_test():
    if not CKPT_PATH.exists():
        print("Checkpoint not found. Train first: python src/models/lstm_model.py")
        sys.exit(1)

    ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
    vocab = ckpt["vocab"]
    model = BiLSTM(len(vocab)).to(DEVICE)
    model.load_state_dict(ckpt["state"])
    model.eval()

    df = load_test()
    dl = DataLoader(TestDataset(df["clean"].values, vocab), batch_size=64)

    all_preds = []
    with torch.no_grad():
        for x in dl:
            preds = (torch.sigmoid(model(x.to(DEVICE))) >= 0.5).long().cpu().numpy()
            all_preds.extend(preds)

    submission = pd.DataFrame({"id": df["id"], "label": all_preds})
    out = ROOT / "submission.csv"
    submission.to_csv(out, index=False)
    print(f"submission.csv saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()
    predict_test() if args.predict else train()
