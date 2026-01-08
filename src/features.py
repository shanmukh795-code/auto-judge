from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Create TF-IDF object (NOT fitted here)
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

def math_symbol_count(text):
    return sum(text.count(s) for s in "+-*/=%<>")

def keyword_features(text):
    keywords = ["graph", "dp", "recursion", "tree", "greedy"]
    return [text.count(k) for k in keywords]

def extra_features(text_series):
    text_len = text_series.apply(len).values.reshape(-1, 1)
    math_cnt = text_series.apply(math_symbol_count).values.reshape(-1, 1)
    kw = np.array(text_series.apply(keyword_features).tolist())
    return np.hstack([text_len, math_cnt, kw])

