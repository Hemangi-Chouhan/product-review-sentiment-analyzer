"""Text cleaning shared by training script and Streamlit app."""
import re

HTML_TAG = re.compile(r"<[^>]+>")
NON_LETTERS = re.compile(r"[^a-z\s']")
MULTI_SPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip HTML tags (reviews contain <br />), remove punctuation/digits."""
    text = str(text).lower()
    text = HTML_TAG.sub(" ", text)
    text = NON_LETTERS.sub(" ", text)
    text = MULTI_SPACE.sub(" ", text).strip()
    return text
