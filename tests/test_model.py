import os
import sys
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.train_model import (
    URL,
    features,
    normalize_columns,
    categorize_position,
)

MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "model.joblib")


def load_prepared_data():
    df = pd.read_csv(URL)
    df = normalize_columns(df)

    df["Position_Group"] = df["Position"].apply(categorize_position)
    df = df.dropna(subset=["Position_Group"])
    df = df[features + ["Position_Group"]].copy()

    X = df[features]
    y = df["Position_Group"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), (
        f"Arquivo não encontrado em: {MODEL_PATH}. "
        "Execute 'python src/train_model.py' antes dos testes."
    )


def test_model_predicts_valid_class():
    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame([{
        "SprintSpeed": 78,
        "Finishing": 80,
        "ShortPassing": 74,
        "Vision": 70,
        "Marking": 35,
        "StandingTackle": 30
    }])

    prediction = model.predict(sample)[0]

    assert prediction in {"Zagueiro", "Meio-campista", "Atacante"}


def test_model_minimum_accuracy():
    model = joblib.load(MODEL_PATH)

    _, X_test, _, y_test = load_prepared_data()
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    assert acc >= 0.70, (
        f"Acurácia abaixo do mínimo esperado. "
        f"Obtida: {acc:.4f}, mínimo esperado: 0.70"
    )