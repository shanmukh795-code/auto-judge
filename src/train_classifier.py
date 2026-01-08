import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from load_data import load_data
from preprocess import preprocess_dataframe

# ---------------- LOAD DATA ----------------

df = load_data("data/problems_data.jsonl")
df = preprocess_dataframe(df)

# ---------------- TF-IDF FEATURES (ONLY THIS) ----------------

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X = tfidf.fit_transform(df["full_text"])

# ---------------- LABELS ----------------

le = LabelEncoder()
y = le.fit_transform(df["problem_class"])

# ---------------- TRAIN-TEST SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# ---------------- EVALUATION ----------------

pred = clf.predict(X_test)
print("Classification Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

# ---------------- SAVE MODELS ----------------

joblib.dump(clf, "models/classifier.pkl")
joblib.dump(tfidf, "models/tfidf.pkl")
joblib.dump(le, "models/label_encoder.pkl")

print("Classifier, TF-IDF, and LabelEncoder saved successfully.")
