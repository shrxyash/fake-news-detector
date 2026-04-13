# %% [markdown]
# # Exploratory Data Analysis — Fake News Dataset

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path("..").resolve()))
from src.preprocess import clean_text

df = pd.read_csv("../data/train.csv")
df["full_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
df.head(3)

# %% [markdown]
# ## Label Distribution

# %%
fig, ax = plt.subplots(figsize=(5, 3))
counts = df["label"].value_counts().sort_index()
counts.index = ["Real (0)", "Fake (1)"]
ax.bar(counts.index, counts.values, color=["#70a0d0", "#e07070"])
for i, v in enumerate(counts.values):
    ax.text(i, v + 1, str(v), ha="center")
ax.set_title("Label Distribution")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Article Length Distribution

# %%
df["word_count"] = df["full_text"].apply(lambda x: len(str(x).split()))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, label, name, color in [(axes[0], 1, "Fake", "#e07070"), (axes[1], 0, "Real", "#70a0d0")]:
    d = df[df["label"] == label]["word_count"]
    ax.hist(d, bins=30, color=color, edgecolor="white")
    ax.axvline(d.median(), color="black", linestyle="--", label=f"Median: {d.median():.0f}")
    ax.set_title(f"{name} — word count"); ax.legend()
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Top Words by Label

# %%
from sklearn.feature_extraction.text import CountVectorizer

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, label, name, color in [(axes[0], 1, "Fake", "#e07070"), (axes[1], 0, "Real", "#70a0d0")]:
    corpus = df[df["label"] == label]["full_text"].apply(clean_text).tolist()
    vec = CountVectorizer(max_features=15, stop_words="english")
    X = vec.fit_transform(corpus)
    counts = sorted(zip(vec.get_feature_names_out(), X.sum(axis=0).A1), key=lambda x: -x[1])
    words, vals = zip(*counts)
    ax.barh(list(words)[::-1], list(vals)[::-1], color=color)
    ax.set_title(f"Top words — {name}")
plt.tight_layout(); plt.show()
