import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report

URL = "https://raw.githubusercontent.com/KeithGalli/matplotlib_tutorial/master/fifa_data.csv"

features = [
    "SprintSpeed",
    "Finishing",
    "ShortPassing",
    "Vision",
    "Marking",
    "StandingTackle"
]


def normalize_columns(df):
    df.columns = df.columns.str.replace(" ", "").str.strip()
    return df


def categorize_position(pos):
    if pd.isna(pos):
        return np.nan

    pos = str(pos).upper()

    if "GK" in pos:
        return np.nan
    if any(p in pos for p in ["CB", "LB", "RB", "LWB", "RWB"]):
        return "Zagueiro"
    if any(p in pos for p in ["CM", "CDM", "CAM", "LM", "RM", "LW", "RW"]):
        return "Meio-campista"
    return "Atacante"


def main():
    df = pd.read_csv(URL)
    df = normalize_columns(df)

    if "Position" not in df.columns:
        raise ValueError("Coluna 'Position' não encontrada no dataset.")

    df["Position_Group"] = df["Position"].apply(categorize_position)
    df = df.dropna(subset=["Position_Group"])

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Features ausentes no dataset: {missing}")

    df = df[features + ["Position_Group"]].copy()

    X = df[features]
    y = df["Position_Group"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), features)
    ])

    models = {
        "KNN": Pipeline([
            ("pre", preprocessor),
            ("model", KNeighborsClassifier())
        ]),
        "DecisionTree": Pipeline([
            ("pre", preprocessor),
            ("model", DecisionTreeClassifier(random_state=42))
        ]),
        "NaiveBayes": Pipeline([
            ("pre", preprocessor),
            ("model", GaussianNB())
        ]),
        "SVM": Pipeline([
            ("pre", preprocessor),
            ("model", SVC())
        ])
    }

    param_grids = {
        "KNN": {
            "model__n_neighbors": [3, 5, 7, 9],
            "model__weights": ["uniform", "distance"]
        },
        "DecisionTree": {
            "model__max_depth": [None, 5, 10, 15],
            "model__min_samples_split": [2, 5, 10]
        },
        "NaiveBayes": {},
        "SVM": {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"]
        }
    }

    best_model = None
    best_score = -1
    best_name = None

    for name, pipeline in models.items():
        print(f"\nTreinando {name}...")

        if param_grids[name]:
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grids[name],
                cv=5,
                scoring="accuracy",
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            print("Melhores hiperparâmetros:", grid.best_params_)
        else:
            model = pipeline.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print(f"Acurácia ({name}): {acc:.4f}")
        print(classification_report(y_test, preds))

        if acc > best_score:
            best_score = acc
            best_score = acc
            best_model = model
            best_name = name

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(best_model, "artifacts/model.joblib")

    print("\n=========================")
    print(f"Melhor modelo: {best_name}")
    print(f"Acurácia final: {best_score:.4f}")
    print("Modelo salvo em artifacts/model.joblib")


if __name__ == "__main__":
    main()