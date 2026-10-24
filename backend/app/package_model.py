"""
Repackage the root-level trained LightGBM model into the format
the backend scoring module expects (backend/artifacts/best_model.joblib).

Run once after train_model.py:
    python -m backend.app.package_model
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_MODEL = ROOT / "artifacts" / "lightgbm_model.joblib"
DST_DIR = ROOT / "backend" / "artifacts"
DST_MODEL = DST_DIR / "best_model.joblib"
FEATURE_LIST = ROOT / "artifacts" / "model_feature_list.json"
METRICS_FILE = ROOT / "artifacts" / "metrics.json"
SHAP_FILE = ROOT / "artifacts" / "shap_global_importance.csv"


def main():
    import json

    if not SRC_MODEL.exists():
        raise FileNotFoundError(
            f"Trained model not found at {SRC_MODEL}. "
            "Run train_model.py first."
        )

    model = joblib.load(SRC_MODEL)

    with open(FEATURE_LIST) as f:
        features = json.load(f)

    with open(METRICS_FILE) as f:
        metrics = json.load(f)

    # Read SHAP global importance for the explanation layer.
    shap_importance = {}
    if SHAP_FILE.exists():
        df = pd.read_csv(SHAP_FILE)
        shap_importance = dict(
            zip(df["feature"], df["mean_absolute_shap"])
        )

    # Choose a threshold from the validation analysis if available.
    threshold_file = ROOT / "artifacts" / "validation_threshold_analysis.csv"
    threshold = 0.30  # default for the demo
    if threshold_file.exists():
        tdf = pd.read_csv(threshold_file)
        # Pick threshold with best F1 on validation
        best = tdf.loc[tdf["f1"].idxmax()]
        threshold = float(best["threshold"])

    bundle = {
        "model": model,
        "model_name": "LightGBM (cash-flow features)",
        "features": features,
        "threshold_from_validation": threshold,
        "shap_global_importance": shap_importance,
        "metrics": {
            "roc_auc": metrics.get("roc_auc"),
            "accuracy": metrics.get("accuracy"),
            "f1": metrics.get("f1"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
        },
        "synthetic_only": True,
    }

    DST_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, DST_MODEL)
    print(f"Packaged model saved to {DST_MODEL}")
    print(f"  Threshold: {threshold}")
    print(f"  Features:  {len(features)}")
    print(f"  SHAP keys: {len(shap_importance)}")


if __name__ == "__main__":
    main()
