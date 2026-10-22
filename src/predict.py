"""Score a CSV of transactions with the trained model.

    python -m src.predict --input data/sample_transactions.csv --output predictions.csv
"""
import argparse
import joblib
import pandas as pd
from src.features import prepare


def load_artifact(path="models/best_model.joblib"):
    return joblib.load(path)


def score_dataframe(df: pd.DataFrame, art: dict) -> pd.DataFrame:
    X, _ = prepare(df, art["scaler"])
    proba = art["model"].predict_proba(X)[:, 1]
    out = df.copy()
    out["fraud_score"] = proba
    out["is_fraud_flag"] = (proba >= art["threshold"]).astype(int)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--model", default="models/best_model.joblib")
    a = ap.parse_args()
    res = score_dataframe(pd.read_csv(a.input), load_artifact(a.model))
    res.to_csv(a.output, index=False)
    print(f"Flagged {res.is_fraud_flag.sum()} of {len(res)} transactions -> {a.output}")
