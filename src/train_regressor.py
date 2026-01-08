import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from load_data import load_data

# ---------------- LOAD DATA ----------------

df = load_data("data/problems_data.jsonl")

# Combine text fields (NO preprocess_text)
df["text"] = (
    df["description"].fillna("") + " " +
    df["input_description"].fillna("") + " " +
    df["output_description"].fillna("")
)

# Target variable (difficulty score)
y = df["problem_score"]

# ---------------- LOAD EXISTING TF-IDF ----------------
# IMPORTANT: use same TF-IDF as classifier

tfidf = joblib.load("models/tfidf.pkl")
X = tfidf.transform(df["text"])

# ---------------- TRAIN-TEST SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- REGRESSION MODEL ----------------

reg = LinearRegression()
reg.fit(X_train, y_train)

# ---------------- EVALUATION ----------------

pred = reg.predict(X_test)

print("Regression MAE:", mean_absolute_error(y_test, pred))

mse = mean_squared_error(y_test, pred)
rmse = mse ** 0.5
print("Regression RMSE:", rmse)

# ---------------- SAVE MODEL ----------------

joblib.dump(reg, "models/regressor.pkl")
print("Regression model saved as models/regressor.pkl")
