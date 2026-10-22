"""End-to-end training pipeline for credit card fraud detection.

    python -m src.train --data data/creditcard.csv

Steps: load -> EDA plots -> split -> features -> train 5 models ->
tune decision threshold on validation set -> evaluate on untouched test set ->
save plots, metrics and the best model.
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.features import FEATURES, prepare

SEED = 42
FIG_DIR = Path("reports/figures")
MODEL_DIR = Path("models")


# ----------------------------------------------------------------- EDA
def run_eda(df: pd.DataFrame):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    counts = df["Class"].value_counts()
    print(f"\nRows: {len(df):,} | Fraud: {counts.get(1, 0):,} "
          f"({df['Class'].mean():.3%})")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.countplot(x="Class", data=df, ax=ax[0])
    ax[0].set_yscale("log")
    ax[0].set_title("Class distribution (log scale)")
    sns.boxplot(x="Class", y=np.log1p(df["Amount"]), data=df, ax=ax[1])
    ax[1].set_title("log(1+Amount) by class")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_eda.png", dpi=150)
    plt.close()

    hour = (df["Time"] // 3600) % 24
    rate = df.groupby(hour)["Class"].mean() * 100
    plt.figure(figsize=(8, 4))
    rate.plot(kind="bar")
    plt.ylabel("Fraud rate (%)")
    plt.xlabel("Hour of day")
    plt.title("Fraud rate by hour")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_fraud_by_hour.png", dpi=150)
    plt.close()


# ------------------------------------------------------------- helpers
def score_of(model, X):
    """Fraud score: probability for classifiers, anomaly score for IsolationForest."""
    if isinstance(model, IsolationForest):
        return -model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def best_threshold(y_true, scores):
    """Threshold that maximises F1 on the validation set."""
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    return float(t[np.argmax(f1)])


def evaluate(y_true, scores, thr):
    pred = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "threshold": thr, "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# ---------------------------------------------------------------- main
def main(args):
    MODEL_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    df = pd.read_csv(args.data).drop_duplicates().reset_index(drop=True)
    run_eda(df)

    # 60 / 20 / 20 stratified split: train, validation (threshold + model selection), test (final report)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["Class"], random_state=SEED)
    train_df, val_df = train_test_split(train_df, test_size=0.25, stratify=train_df["Class"], random_state=SEED)

    X_train, scaler = prepare(train_df)          # scaler fit on train only (no leakage)
    X_val, _ = prepare(val_df, scaler)
    X_test, _ = prepare(test_df, scaler)
    y_train, y_val, y_test = train_df["Class"], val_df["Class"], test_df["Class"]

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    print(f"Train {len(train_df):,} | Val {len(val_df):,} | Test {len(test_df):,}")

    # SMOTE oversampling is applied to TRAIN ONLY, for the logistic regression variant
    X_sm, y_sm = SMOTE(random_state=SEED, sampling_strategy=0.1).fit_resample(X_train, y_train)

    models = {
        "LogReg (class_weight)": (LogisticRegression(max_iter=2000, class_weight="balanced"), (X_train, y_train)),
        "LogReg + SMOTE": (LogisticRegression(max_iter=2000), (X_sm, y_sm)),
        "Random Forest": (RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED), (X_train, y_train)),
        "XGBoost": (XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=neg / max(pos, 1),
            eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=SEED), (X_train, y_train)),
        "Isolation Forest (unsupervised)": (IsolationForest(
            n_estimators=200, contamination=float(max(pos / len(y_train), 1e-4)),
            n_jobs=-1, random_state=SEED), (X_train, None)),
    }

    results, fitted, thresholds, test_scores = {}, {}, {}, {}
    for name, (model, (Xf, yf)) in models.items():
        print(f"Training: {name}")
        model.fit(Xf) if yf is None else model.fit(Xf, yf)
        s_val = score_of(model, X_val)
        thr = best_threshold(y_val, s_val)
        s_test = score_of(model, X_test)
        m = evaluate(y_test, s_test, thr)
        m["val_pr_auc"] = average_precision_score(y_val, s_val)
        results[name], fitted[name], thresholds[name], test_scores[name] = m, model, thr, s_test

    table = pd.DataFrame(results).T[
        ["roc_auc", "pr_auc", "precision", "recall", "f1", "tp", "fp", "fn", "tn", "val_pr_auc"]]
    table[["tp", "fp", "fn", "tn"]] = table[["tp", "fp", "fn", "tn"]].astype(int)
    print("\n=== TEST SET RESULTS ===")
    print(table.round(4).to_string())
    table.round(4).to_csv("reports/metrics.csv")

    # Pick best SUPERVISED model by validation PR-AUC (not by test -> no selection bias)
    sup = {k: v["val_pr_auc"] for k, v in results.items() if "Isolation" not in k}
    best = max(sup, key=sup.get)
    print(f"\nBest model (by validation PR-AUC): {best}")

    joblib.dump({"model": fitted[best], "scaler": scaler, "threshold": thresholds[best],
                 "features": FEATURES, "name": best}, MODEL_DIR / "best_model.joblib")
    with open("reports/metrics.json", "w") as f:
        json.dump({"best_model": best, "results": results}, f, indent=2)

    # ---- Plots
    plt.figure(figsize=(7, 5))
    for n, s in test_scores.items():
        p, r, _ = precision_recall_curve(y_test, s)
        plt.plot(r, p, label=f"{n} (AP={results[n]['pr_auc']:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall curves (test)")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(FIG_DIR / "03_pr_curves.png", dpi=150); plt.close()

    plt.figure(figsize=(7, 5))
    for n, s in test_scores.items():
        fpr, tpr, _ = roc_curve(y_test, s)
        plt.plot(fpr, tpr, label=f"{n} (AUC={results[n]['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curves (test)")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(FIG_DIR / "04_roc_curves.png", dpi=150); plt.close()

    r = results[best]
    cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred legit", "Pred fraud"], yticklabels=["Actual legit", "Actual fraud"])
    plt.title(f"Confusion matrix: {best}"); plt.tight_layout()
    plt.savefig(FIG_DIR / "05_confusion_matrix.png", dpi=150); plt.close()

    # ---- Explainability (SHAP) for tree models
    if best in ("XGBoost", "Random Forest"):
        try:
            import shap
            sample = X_test.sample(min(2000, len(X_test)), random_state=SEED)
            sv = shap.TreeExplainer(fitted[best]).shap_values(sample)
            if isinstance(sv, list):
                sv = sv[1]
            elif getattr(sv, "ndim", 2) == 3:
                sv = sv[:, :, 1]
            shap.summary_plot(sv, sample, show=False)
            plt.tight_layout()
            plt.savefig(FIG_DIR / "06_shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()
        except Exception as e:  # SHAP is a nice-to-have, never break training
            print(f"SHAP skipped: {e}")

    # ---- Business view: money caught vs. false alarms on the test set
    t = test_df.copy()
    t["flag"] = (test_scores[best] >= thresholds[best]).astype(int)
    caught = t.loc[(t.Class == 1) & (t.flag == 1), "Amount"].sum()
    total = t.loc[t.Class == 1, "Amount"].sum()
    print(f"Fraud amount caught: {caught:,.0f} of {total:,.0f} ({caught / max(total, 1):.1%}) "
          f"| false alarms: {r['fp']}")
    print("\nDone. See reports/ and models/.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard.csv")
    main(ap.parse_args())
