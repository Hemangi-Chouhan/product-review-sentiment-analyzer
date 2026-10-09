"""Streamlit app: type a product review, get sentiment + confidence + key words."""
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))
from text_utils import clean_text  # noqa: E402

MODEL_PATH = ROOT / "models" / "sentiment_model.joblib"

st.set_page_config(page_title="Amazon Review Sentiment Analyzer", page_icon="🛒")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def explain(pipeline, cleaned: str, top_n: int = 8) -> pd.DataFrame:
    """Words in the review that pushed the prediction most (tfidf value x weight)."""
    vec, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    row = vec.transform([cleaned])
    names = vec.get_feature_names_out()
    contrib = {names[i]: row[0, i] * clf.coef_[0][i] for i in row.nonzero()[1]}
    df = pd.DataFrame(list(contrib.items()), columns=["word", "impact"])
    if df.empty:
        return df
    df = df.reindex(df["impact"].abs().sort_values(ascending=False).index).head(top_n)
    df["pushes toward"] = df["impact"].apply(lambda v: "positive" if v > 0 else "negative")
    return df.reset_index(drop=True)


st.title("🛒 Amazon Review Sentiment Analyzer")
st.caption("TF-IDF + Logistic Regression | trained on Amazon product reviews")

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python src/train.py` first.")
    st.stop()

model = load_model()

examples = {
    "Positive example": "Great taste and fast delivery. Totally worth the price, I will buy again!",
    "Negative example": "Not good at all. Arrived stale and the quality is terrible. Waste of money.",
}
choice = st.radio("Try an example or write your own:", ["Custom", *examples], horizontal=True)
default = examples.get(choice, "")
review = st.text_area("Product review", value=default, height=150)

if st.button("Analyze sentiment", type="primary"):
    cleaned = clean_text(review)
    if not cleaned:
        st.warning("Please enter a review.")
    else:
        proba = model.predict_proba([cleaned])[0]
        pos = proba[1]
        confidence = max(pos, 1 - pos)
        if confidence < 0.6:
            st.info(f"MIXED / UNCERTAIN 😐  ({confidence:.1%} confidence)")
        elif pos >= 0.5:
            st.success(f"POSITIVE 😊  ({confidence:.1%} confidence)")
        else:
            st.error(f"NEGATIVE 😞  ({confidence:.1%} confidence)")
        st.progress(float(pos), text=f"Positive probability: {pos:.1%}")

        exp = explain(model, cleaned)
        if not exp.empty:
            st.subheader("Words that influenced the result")
            st.dataframe(exp[["word", "pushes toward"]], hide_index=True, use_container_width=True)
