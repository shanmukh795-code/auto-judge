from flask import Flask, request
import joblib

app = Flask(__name__)

# ---------------- LOAD MODELS (SAME AS TRAINING) ----------------

clf = joblib.load("models/classifier.pkl")
reg = joblib.load("models/regressor.pkl")
tfidf = joblib.load("models/tfidf.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# ---------------- HTML WITH ATTRACTIVE UI ----------------

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AutoJudge</title>
    <style>
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .card {
            background: white;
            padding: 30px;
            width: 700px;
            border-radius: 12px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            color: #1e3c72;
        }
        p {
            text-align: center;
            color: #555;
        }
        label {
            font-weight: bold;
        }
        textarea {
            width: 100%;
            height: 70px;
            margin-top: 5px;
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 6px;
            border: 1px solid #ccc;
            resize: none;
        }
        button {
            width: 100%;
            padding: 12px;
            background: #1e3c72;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
        }
        button:hover {
            background: #16335d;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            background: #f0f4ff;
            border-radius: 8px;
            font-size: 16px;
            text-align: center;
        }
    </style>
</head>
<body>

<div class="card">
    <h1>AutoJudge</h1>
    <p>Predict Programming Problem Difficulty using Machine Learning</p>

    <form method="post">
        <label>Problem Description</label>
        <textarea name="problem" placeholder="Describe the problem..." required></textarea>

        <label>Input Description</label>
        <textarea name="input" placeholder="Describe the input..." required></textarea>

        <label>Output Description</label>
        <textarea name="output" placeholder="Describe the output..." required></textarea>

        <button type="submit">Predict Difficulty</button>
    </form>

    __RESULT__
</div>

</body>
</html>
"""

# ---------------- FLASK LOGIC ----------------

@app.route("/", methods=["GET", "POST"])
def home():
    result_html = ""

    if request.method == "POST":
        problem = request.form["problem"]
        inp = request.form["input"]
        out = request.form["output"]

        combined_text = problem + " " + inp + " " + out

        features = tfidf.transform([combined_text])

        class_num = clf.predict(features)[0]
        class_name = label_encoder.inverse_transform([class_num])[0]

        score = round(reg.predict(features)[0], 2)

        result_html = f"""
        <div class="result">
            <b>Predicted Difficulty:</b> {class_name}<br>
            <b>Predicted Score:</b> {score}
        </div>
        """

    return HTML_PAGE.replace("__RESULT__", result_html)

# ---------------- RUN APP ----------------

if __name__ == "__main__":
    app.run(debug=False)
