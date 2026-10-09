"""Train a TF-IDF + Logistic Regression sentiment classifier on Amazon product reviews.

Usage:
    python src/train.py
    python src/train.py --data data/Reviews.csv --sample 100000
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from text_utils import clean_text

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "sentiment_model.joblib"
REPORTS = ROOT / "reports"
DEFAULT_DATA_CANDIDATES = [
    ROOT / "data" / "Reviews.csv",
    ROOT.parent / "Reviews.csv",
    ROOT / "Reviews.csv",
    ROOT.parent / "data" / "Reviews.csv",
]


def resolve_data_path(path: Path) -> Path:
    """Accept the dataset from either the project root or a nested data/ folder."""
    if path.exists():
        return path
    for candidate in DEFAULT_DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    for candidate in DEFAULT_DATA_CANDIDATES:
        if candidate.parent.exists():
            raise FileNotFoundError(
                f"Could not find Reviews.csv. Expected '{path}' or one of: "
                + ", ".join(str(p) for p in DEFAULT_DATA_CANDIDATES)
            )
    raise FileNotFoundError(f"Could not find Reviews.csv at '{path}' or in project folders.")


# Keep negation words: "not good" must not become "good"
NEGATIONS = {"not", "no", "nor", "never", "nothing", "cannot", "neither", "none", "nobody"}
STOP_WORDS = sorted(ENGLISH_STOP_WORDS - NEGATIONS)


def load_data(path: Path, sample: int) -> pd.DataFrame:
    """Load Amazon reviews. Supports Kaggle 'Amazon Fine Food Reviews' (Text, Score, Summary)
    and the common alternative (reviewText, overall, summary)."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    text_col = cols.get("text") or cols.get("reviewtext") or cols.get("review")
    score_col = cols.get("score") or cols.get("overall") or cols.get("rating")
    summ_col = cols.get("summary")
    if not text_col or not score_col:
        raise ValueError(f"Could not find review text / rating columns in {list(df.columns)}")

    df = df.dropna(subset=[text_col, score_col])
    df = df[df[score_col] != 3]                       # drop neutral 3-star reviews
    df["label"] = (df[score_col] >= 4).astype(int)    # 4-5 = positive, 1-2 = negative
    full_text = df[text_col].astype(str)
    if summ_col:                                      # the short title often carries sentiment
        full_text = df[summ_col].fillna("").astype(str) + " " + full_text
    df["review"] = full_text
    df = df.drop_duplicates(subset="review")

    if sample and len(df) > sample:                   # keep training fast
        df = df.sample(sample, random_state=42)
    df["clean_review"] = df["review"].apply(clean_text)
    return df[["review", "clean_review", "label"]]


def plot_top_words(pipeline: Pipeline, n: int = 20) -> None:
    vec = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    words = pd.Series(clf.coef_[0], index=vec.get_feature_names_out())
    top_pos = words.nlargest(n)
    top_neg = words.nsmallest(n)

    pd.DataFrame({"word": top_pos.index, "weight": top_pos.values}).to_csv(
        REPORTS / "top_positive_words.csv", index=False)
    pd.DataFrame({"word": top_neg.index, "weight": top_neg.values}).to_csv(
        REPORTS / "top_negative_words.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    top_pos[::-1].plot.barh(ax=axes[0], color="seagreen")
    axes[0].set_title(f"Top {n} words -> POSITIVE")
    top_neg[::-1].plot.barh(ax=axes[1], color="crimson")
    axes[1].set_title(f"Top {n} words -> NEGATIVE")
    for ax in axes:
        ax.set_xlabel("Model weight")
    plt.tight_layout()
    fig.savefig(REPORTS / "top_words.png", dpi=150)
    plt.close(fig)


def main(data_path: Path, sample: int) -> None:
    REPORTS.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    data_path = resolve_data_path(data_path)

    print(f"Loading {data_path} ...")
    df = load_data(data_path, sample)
    print(f"Reviews used: {len(df)} | Positive share: {df['label'].mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_review"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"])

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 2),
                                  min_df=3, max_features=150_000,
                                  sublinear_tf=True)),
        ("clf", LogisticRegression(C=5, max_iter=1000, class_weight="balanced")),
    ])

    print("Training ...")
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    negative_recall = recall_score(y_test, preds, labels=[0], average="macro")
    print(
        f"\nAccuracy: {acc:.4f} | Macro F1: {f1:.4f} | Negative recall: {negative_recall:.4f}\n"
    )
    print(classification_report(y_test, preds, target_names=["negative", "positive"]))

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds, display_labels=["negative", "positive"], cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    plot_top_words(pipeline)

    (REPORTS / "metrics.json").write_text(json.dumps(
        {"accuracy": round(acc, 4), "macro_f1": round(f1, 4),
         "negative_recall": round(negative_recall, 4),
         "train_size": len(X_train), "test_size": len(X_test)}, indent=2))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved charts/metrics -> {REPORTS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "Reviews.csv"),
                        help="Path to Reviews.csv; defaults to the project data folder")
    parser.add_argument("--sample", type=int, default=100000,
                        help="max reviews to use (0 = all)")
    args = parser.parse_args()
    main(Path(args.data), args.sample)
