import os
import joblib
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "model.joblib")

model = joblib.load(MODEL_PATH)

FEATURES = [
    "SprintSpeed",
    "Finishing",
    "ShortPassing",
    "Vision",
    "Marking",
    "StandingTackle"
]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            values = {field: float(request.form[field]) for field in FEATURES}
            data = pd.DataFrame([values], columns=FEATURES)
            prediction = model.predict(data)[0]
        except Exception as e:
            error = f"Não foi possível processar a predição. Detalhe: {e}"

    return render_template("index.html", prediction=prediction, error=error)


if __name__ == "__main__":
    app.run(debug=True, port=5001)