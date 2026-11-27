import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from ..data import load_json


def download_nltk_data():
    for resource in ("stopwords", "wordnet", "omw-1.4"):
        nltk.download(resource, quiet=True, raise_on_error=True)


@lru_cache(maxsize=1)
def english_stop_words():
    return set(stopwords.words("english")) - {"no", "not", "nor", "never", "without"}


LEMMATIZER = WordNetLemmatizer()


def tokenize_text(text):
    """BM25 preprocessing: lowercase, tokenize, remove stopwords, lemmatize."""
    words = re.findall(r"\w+", str(text).lower())
    stop_words = english_stop_words()
    return [LEMMATIZER.lemmatize(word) for word in words if word not in stop_words]


def load_evidence_corpus(evidence_path):
    """Load evidence and preprocess it once for BM25."""
    evidence = load_json(evidence_path)
    if not evidence:
        raise ValueError(f"Evidence corpus is empty: {evidence_path}")
    ids = list(evidence)
    texts = list(evidence.values())

    return {
        "ids": ids,
        "texts": texts,
        "tokens": [tokenize_text(text) for text in texts],
        "id_to_pos": {eid: pos for pos, eid in enumerate(ids)},
    }
