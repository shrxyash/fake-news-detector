import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = set()

DATA_DIR = Path(__file__).parent.parent / "data"


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if remove_stopwords and STOPWORDS:
        text = " ".join(w for w in text.split() if w not in STOPWORDS and len(w) > 1)
    return text


def load_train(path: str = None) -> pd.DataFrame:
    path = path or str(DATA_DIR / "train.csv")
    df = pd.read_csv(path)
    df["title"] = df["title"].fillna("")
    df["text"]  = df["text"].fillna("")
    df["full_text"] = df["title"] + " " + df["text"]
    df["clean"] = df["full_text"].apply(clean_text)
    print(f"Loaded {len(df)} training articles | fake={df['label'].sum()} real={(df['label']==0).sum()}")
    return df


def load_test(path: str = None) -> pd.DataFrame:
    path = path or str(DATA_DIR / "test.csv")
    df = pd.read_csv(path)
    df["title"] = df["title"].fillna("")
    df["text"]  = df["text"].fillna("")
    df["full_text"] = df["title"] + " " + df["text"]
    df["clean"] = df["full_text"].apply(clean_text)
    print(f"Loaded {len(df)} test articles")
    return df


def get_splits(val_size: float = 0.15):
    df = load_train()
    X, y = df["clean"].values, df["label"].values
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=42, stratify=y
    )
    print(f"Split → train={len(X_train)} | val={len(X_val)}")
    return X_train, X_val, y_train, y_val
