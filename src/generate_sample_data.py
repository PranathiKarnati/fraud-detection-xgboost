"""Generate a SYNTHETIC dataset with the same schema as the Kaggle credit card
fraud data. Use it only to test the pipeline; use the real data for results.

    python -m src.generate_sample_data --rows 30000
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def main(rows: int, out: str):
    X, y = make_classification(
        n_samples=rows, n_features=28, n_informative=12, n_redundant=4,
        weights=[0.99, 0.01], class_sep=1.2, flip_y=0.002, random_state=42,
    )
    rng = np.random.default_rng(42)
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.sort(rng.integers(0, 172_800, rows)))
    df["Amount"] = np.round(rng.lognormal(3.2, 1.4, rows), 2)
    df["Class"] = y
    df.to_csv(out, index=False)
    print(f"Saved {rows} rows to {out} | fraud rate: {y.mean():.3%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=30000)
    p.add_argument("--out", default="data/creditcard.csv")
    a = p.parse_args()
    main(a.rows, a.out)
