"""Train the three ML models on real datasets and save them with honest metrics.

Run:  python train_models.py
Output: models/*.joblib and models/metrics.json
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
SEED = 42


def ranges(df, cols):
    """1st-99th percentile of each input, so the app can warn on extrapolation."""
    return {c: [float(df[c].quantile(0.01)), float(df[c].quantile(0.99))] for c in cols}


def regression_metrics(model, X_tr, X_te, y_tr, y_te, baseline):
    baseline.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return {
        "r2": round(float(r2_score(y_te, pred)), 3),
        "mae": round(float(mean_absolute_error(y_te, pred)), 2),
        "baseline_mae": round(float(mean_absolute_error(y_te, baseline.predict(X_te))), 2),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
    }


def train_house():
    df = pd.read_csv(DATA / "house.csv")
    df = df[(df.bedrooms.between(1, 10)) & (df.bathrooms > 0)]  # drops data-entry junk (e.g. 33 bedrooms)
    feats = ["sqft_living", "bedrooms", "bathrooms"]
    X_tr, X_te, y_tr, y_te = train_test_split(df[feats], df["price"], test_size=0.2, random_state=SEED)
    model = TransformedTargetRegressor(
        regressor=HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=SEED),
        func=np.log1p,
        inverse_func=np.expm1,
    ).fit(X_tr, y_tr)
    m = regression_metrics(model, X_tr, X_te, y_tr, y_te, DummyRegressor(strategy="median"))
    m["ranges"] = ranges(df, feats)
    joblib.dump(model, MODELS / "house_model.joblib")
    return m


def train_insurance():
    df = pd.read_csv(DATA / "insurance.csv")
    feats = ["age", "sex", "bmi", "children", "smoker", "region"]
    X_tr, X_te, y_tr, y_te = train_test_split(df[feats], df["charges"], test_size=0.2, random_state=SEED)
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])],
        remainder="passthrough",
    )
    model = Pipeline([
        ("pre", pre),
        ("gb", HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=3,
                                             min_samples_leaf=20, random_state=SEED)),
    ]).fit(X_tr, y_tr)
    m = regression_metrics(model, X_tr, X_te, y_tr, y_te, DummyRegressor(strategy="median"))
    m["ranges"] = ranges(df, ["age", "bmi", "children"])
    joblib.dump(model, MODELS / "insurance_model.joblib")
    return m


def train_titanic():
    df = pd.read_csv(DATA / "titanic.csv")
    feats = ["pclass", "sex", "age"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[feats], df["survived"], test_size=0.2, random_state=SEED, stratify=df["survived"]
    )
    pre = ColumnTransformer(
        [
            ("sex", OneHotEncoder(handle_unknown="ignore"), ["sex"]),  # strings in, so no manual 0/1 mapping to get wrong
            ("age", SimpleImputer(strategy="median"), ["age"]),
        ],
        remainder="passthrough",
    )
    model = Pipeline([
        ("pre", pre),
        ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=SEED)),
    ]).fit(X_tr, y_tr)
    base = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    joblib.dump(model, MODELS / "titanic_model.joblib")
    return {
        "accuracy": round(float(accuracy_score(y_te, model.predict(X_te))), 3),
        "baseline_accuracy": round(float(accuracy_score(y_te, base.predict(X_te))), 3),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
    }


def main():
    MODELS.mkdir(exist_ok=True)
    metrics = {"house": train_house(), "insurance": train_insurance(), "titanic": train_titanic()}
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
