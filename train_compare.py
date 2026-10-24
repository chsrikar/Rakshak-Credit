"""Compare tabular ML models for Rakshak Credit using existing fixed splits.

Run from the project root:
    python train_compare.py --data-dir data/data --out-dir artifacts_comparison

Synthetic labels only: results measure agreement with the generated toy labels,
not real-world credit default performance.
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, brier_score_loss, confusion_matrix)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "synthetic_missed_payment"

def load_features(data_dir: Path, train: pd.DataFrame):
    p = data_dir / "model_feature_list.json"
    if p.exists():
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, list): return [x for x in obj if x in train.columns and x != TARGET]
        if isinstance(obj, dict):
            for k in ("features", "model_features", "feature_list"):
                if isinstance(obj.get(k), list):
                    return [x for x in obj[k] if x in train.columns and x != TARGET]
    return [c for c in train.select_dtypes(include=[np.number]).columns if c != TARGET and not c.startswith("Unnamed:")]

def safe_auc(fn, y, p):
    try: return float(fn(y, p))
    except Exception: return float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/data")
    ap.add_argument("--out-dir", default="artifacts_comparison")
    args = ap.parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {s: data_dir / f"{s}.csv" for s in ("train", "validation", "test")}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing: raise FileNotFoundError("Missing split files: " + ", ".join(missing))
    frames = {s: pd.read_csv(p) for s,p in paths.items()}
    for s, df in frames.items():
        if TARGET not in df.columns: raise ValueError(f"{TARGET} not found in {paths[s]}")
    features = load_features(data_dir, frames["train"])
    if not features: raise ValueError("No model features found. Check model_feature_list.json and split CSVs.")
    X, y = {}, {}
    for s, df in frames.items():
        X[s] = df.reindex(columns=features).replace([np.inf, -np.inf], np.nan)
        y[s] = df[TARGET].astype(int)
    models = {
        "LogisticRegression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
        "RandomForest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=500, min_samples_leaf=3, class_weight="balanced", max_features="sqrt", n_jobs=-1, random_state=42)),
        "ExtraTrees": make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3, class_weight="balanced", max_features="sqrt", n_jobs=-1, random_state=42)),
        "HistGradientBoosting": make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, l2_regularization=2.0, max_leaf_nodes=15, random_state=42)),
        "GradientBoosting": make_pipeline(SimpleImputer(strategy="median"), GradientBoostingClassifier(n_estimators=150, learning_rate=0.04, max_depth=2, min_samples_leaf=10, random_state=42)),
    }
    rows, pred_rows = [], []
    for name, model in models.items():
        print(f"\nTraining {name}...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X["train"], y["train"])
        pv = model.predict_proba(X["validation"])[:, 1]
        # Pick a threshold using validation only; never tune on the test set.
        candidates = np.arange(0.10, 0.701, 0.01)
        vals = [(f1_score(y["validation"], pv >= t, zero_division=0), t) for t in candidates]
        best_f1 = max(v[0] for v in vals)
        threshold = min(t for f,t in vals if f >= best_f1 - 1e-12)
        pt = model.predict_proba(X["test"])[:, 1]
        yh = (pt >= threshold).astype(int)
        cm = confusion_matrix(y["test"], yh, labels=[0,1])
        row = {
            "model": name, "features": len(features), "threshold_from_validation": round(float(threshold), 3),
            "validation_f1_at_selected_threshold": round(float(best_f1), 4),
            "test_roc_auc": round(safe_auc(roc_auc_score, y["test"], pt), 4),
            "test_pr_auc_average_precision": round(safe_auc(average_precision_score, y["test"], pt), 4),
            "test_accuracy": round(float(accuracy_score(y["test"], yh)), 4),
            "test_precision": round(float(precision_score(y["test"], yh, zero_division=0)), 4),
            "test_recall": round(float(recall_score(y["test"], yh, zero_division=0)), 4),
            "test_f1": round(float(f1_score(y["test"], yh, zero_division=0)), 4),
            "test_brier_score": round(float(brier_score_loss(y["test"], pt)), 4),
            "test_tn": int(cm[0,0]), "test_fp": int(cm[0,1]), "test_fn": int(cm[1,0]), "test_tp": int(cm[1,1]),
        }
        rows.append(row)
        pred_rows.extend({"model": name, "row_index": int(i), "actual": int(y["test"].iloc[i]), "probability": float(pt[i]), "predicted": int(yh[i]), "threshold": float(threshold)} for i in range(len(pt)))
        dump(model, out_dir / f"{name.lower()}_model.joblib")
        print(f"  ROC-AUC={row['test_roc_auc']:.3f} | PR-AUC={row['test_pr_auc_average_precision']:.3f} | Recall={row['test_recall']:.3f} | F1={row['test_f1']:.3f} | threshold={threshold:.2f}")
    results = pd.DataFrame(rows).sort_values(["test_roc_auc", "test_pr_auc_average_precision"], ascending=False)
    results.to_csv(out_dir / "model_comparison.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out_dir / "test_predictions_all_models.csv", index=False)
    (out_dir / "run_info.json").write_text(json.dumps({"synthetic_only": True, "target": TARGET, "features": features, "train_rows": len(frames["train"]), "validation_rows": len(frames["validation"]), "test_rows": len(frames["test"]), "note": "Threshold selected on validation split. Test metrics evaluate synthetic labels only."}, indent=2), encoding="utf-8")
    print("\n=== MODEL COMPARISON (sorted by test ROC-AUC) ===")
    print(results[["model", "test_roc_auc", "test_pr_auc_average_precision", "test_precision", "test_recall", "test_f1", "threshold_from_validation"]].to_string(index=False))
    print(f"\nSaved results to: {out_dir.resolve()}")

if __name__ == "__main__": main()
