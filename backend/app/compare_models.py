"""
Step 4G — Model Comparison: LightGBM Baseline vs. Deep Tabular Neural Network
Evaluates both models on the held-out test dataset (225 rows).
Computes consistent metrics (ROC-AUC, Accuracy, Precision, Recall, F1, Brier Score, Review Rate).
Saves output to artifacts/model_comparison.json.
"""
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
)
try:
    from .scoring import _load_bundle, FEATURES
    from .neural_scoring import load_neural_model, neural_model_score
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.scoring import _load_bundle, FEATURES
    from app.neural_scoring import load_neural_model, neural_model_score

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
BACKEND_ARTIFACTS = ROOT / "backend" / "artifacts"
OUTPUT_FILE = ARTIFACTS_DIR / "model_comparison.json"


def _distribution(values: np.ndarray, threshold: float) -> dict:
    """Compact score-distribution diagnostics for benchmark auditability."""
    return {
        "min": round(float(np.min(values)), 6),
        "p10": round(float(np.quantile(values, 0.10)), 6),
        "median": round(float(np.median(values)), 6),
        "p90": round(float(np.quantile(values, 0.90)), 6),
        "max": round(float(np.max(values)), 6),
        "below_threshold": int(np.sum(values < threshold)),
        "at_or_above_threshold": int(np.sum(values >= threshold)),
    }


def build_77_feature_dict(test_row: pd.Series, applicant_row: pd.Series, template: dict) -> dict:
    """Map test set features and applicant demographic context into the 77 neural features."""
    feat = template.copy()

    # Demographics & Context
    if applicant_row is not None and not applicant_row.empty:
        feat["occupation"] = str(applicant_row.get("occupation_type", "kirana_shopkeeper"))
        feat["city"] = str(applicant_row.get("city", "Kochi"))
        feat["state"] = str(applicant_row.get("state", "Kerala"))
        feat["loan_purpose"] = str(applicant_row.get("loan_purpose_category", "inventory_purchase"))

    # Financial and Cash-Flow Metrics from test.csv
    monthly_inflow = float(test_row.get("avg_monthly_income", 40000.0))
    monthly_expenses = float(test_row.get("avg_monthly_expenses", 25000.0))
    existing_debt = float(test_row.get("self_reported_existing_monthly_debt_inr", 3000.0))
    loan_amount = float(test_row.get("requested_loan_amount_inr", 50000.0))
    tenure = float(test_row.get("requested_tenure_months", 12.0))
    annual_rate = 18.0

    r = (annual_rate / 100.0) / 12.0
    if r > 0 and tenure > 0:
        emi = loan_amount * r * ((1.0 + r) ** tenure) / (((1.0 + r) ** tenure) - 1.0)
    else:
        emi = loan_amount / max(1.0, tenure)

    surplus_before = monthly_inflow - monthly_expenses - existing_debt
    surplus_after = surplus_before - emi

    feat["requested_loan_amount_inr"] = loan_amount
    feat["loan_tenure_months"] = int(tenure)
    feat["annual_interest_rate_pct"] = annual_rate
    feat["monthly_emi_inr"] = round(emi, 2)
    feat["monthly_inflow_inr"] = round(monthly_inflow, 2)
    feat["monthly_expenses_inr"] = round(monthly_expenses, 2)
    feat["existing_monthly_debt_inr"] = round(existing_debt, 2)
    feat["net_monthly_surplus_before_new_emi_inr"] = round(surplus_before, 2)
    feat["net_monthly_surplus_after_new_emi_inr"] = round(surplus_after, 2)

    feat["emi_to_inflow_ratio"] = round(emi / max(1.0, monthly_inflow), 4)
    feat["total_debt_service_ratio"] = round((existing_debt + emi) / max(1.0, monthly_inflow), 4)
    feat["expense_to_inflow_ratio"] = round(monthly_expenses / max(1.0, monthly_inflow), 4)
    feat["loan_to_monthly_inflow_ratio"] = round(loan_amount / max(1.0, monthly_inflow), 4)

    feat["income_regularity_score"] = float(test_row.get("transaction_regularity", 0.85))
    feat["inflow_volatility_cv"] = float(test_row.get("income_cv", 0.25))
    feat["income_stability_index"] = float(test_row.get("positive_income_months_ratio", 0.90))
    feat["income_source_concentration"] = float(test_row.get("income_source_concentration", 0.50))
    feat["emergency_buffer_months"] = round(float(test_row.get("savings_ratio", 0.20)) * 6.0, 2)
    feat["reversal_or_failed_txn_rate"] = float(test_row.get("low_balance_days_ratio", 0.05))

    feat["affordability_stress_flag"] = 1 if surplus_after < 0 else 0
    feat["repayment_capacity_band"] = "strong" if surplus_after > emi * 1.5 else ("moderate" if surplus_after > 0 else "stressed")

    return feat


def run_comparison(output_file: Path = OUTPUT_FILE):
    print("=" * 70)
    print("RAKSHAK CREDIT — MODEL BENCHMARK & COMPARISON")
    print("Evaluating LightGBM Baseline vs. Deep Tabular Neural Network")
    print("=" * 70)

    # 1. Load test data
    test_path = DATA_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found at {test_path}")

    test_df = pd.read_csv(test_path)
    y_true = test_df["synthetic_missed_payment"].values
    n_samples = len(test_df)
    n_positive = int(np.sum(y_true))
    print(f"Loaded held-out test set: {n_samples} samples ({n_positive} positive defaults, rate: {n_positive/n_samples:.2%})")

    # 2. Evaluate LightGBM Baseline
    bundle = _load_bundle()
    if bundle is None:
        raise FileNotFoundError("LightGBM model bundle not found.")

    lgb_model = bundle["model"]
    lgb_threshold = float(bundle.get("threshold_from_validation", 0.30))
    X_lgb = test_df[FEATURES]
    lgb_probs = lgb_model.predict_proba(X_lgb)[:, 1]
    lgb_preds = (lgb_probs >= lgb_threshold).astype(int)

    lgb_auc = roc_auc_score(y_true, lgb_probs)
    lgb_acc = accuracy_score(y_true, lgb_preds)
    lgb_prec = precision_score(y_true, lgb_preds, zero_division=0)
    lgb_rec = recall_score(y_true, lgb_preds, zero_division=0)
    lgb_f1 = f1_score(y_true, lgb_preds, zero_division=0)
    lgb_brier = brier_score_loss(y_true, lgb_probs)
    lgb_review_rate = float(np.mean(lgb_preds))

    # 3. Evaluate Deep Tabular Neural Network
    load_neural_model()
    template_path = ROOT / "backend" / "data" / "neural_sample_payload.json"
    template = json.loads(template_path.read_text(encoding="utf-8"))["features"]

    applicants_path = DATA_DIR / "applicants.csv"
    applicants_df = pd.read_csv(applicants_path).set_index("applicant_id") if applicants_path.exists() else None

    neural_probs = []
    neural_preds = []
    neural_threshold = float(__import__("app.neural_scoring", fromlist=["_metadata"])._metadata.get("review_threshold", 0.28))

    for _, row in test_df.iterrows():
        app_id = row.get("applicant_id")
        app_row = applicants_df.loc[app_id] if applicants_df is not None and app_id in applicants_df.index else None
        feat_dict = build_77_feature_dict(row, app_row, template)
        score_res = neural_model_score(feat_dict)
        prob = score_res["risk_score"]
        neural_probs.append(prob)
        neural_preds.append(1 if prob >= neural_threshold else 0)

    neural_probs = np.array(neural_probs)
    neural_preds = np.array(neural_preds)

    nn_auc = roc_auc_score(y_true, neural_probs)
    nn_acc = accuracy_score(y_true, neural_preds)
    nn_prec = precision_score(y_true, neural_preds, zero_division=0)
    nn_rec = recall_score(y_true, neural_preds, zero_division=0)
    nn_f1 = f1_score(y_true, neural_preds, zero_division=0)
    nn_brier = brier_score_loss(y_true, neural_probs)
    nn_review_rate = float(np.mean(neural_preds))

    # 4. Display Comparison Table
    print("\nBenchmark Results Summary:")
    print("-" * 70)
    print(f"{'Metric':<25} | {'LightGBM Baseline':<18} | {'Deep Tabular Net':<18} | {'Delta':<10}")
    print("-" * 70)
    print(f"{'Feature Count':<25} | {len(FEATURES):<18} | {77:<18} | {'+51 features':<10}")
    print(f"{'Decision Threshold':<25} | {lgb_threshold:<18.2f} | {neural_threshold:<18.2f} | {'-':<10}")
    print(f"{'ROC-AUC':<25} | {lgb_auc:<18.4f} | {nn_auc:<18.4f} | {nn_auc - lgb_auc:+10.4f}")
    print(f"{'Accuracy':<25} | {lgb_acc:<18.4f} | {nn_acc:<18.4f} | {nn_acc - lgb_acc:+10.4f}")
    print(f"{'Precision':<25} | {lgb_prec:<18.4f} | {nn_prec:<18.4f} | {nn_prec - lgb_prec:+10.4f}")
    print(f"{'Recall':<25} | {lgb_rec:<18.4f} | {nn_rec:<18.4f} | {nn_rec - lgb_rec:+10.4f}")
    print(f"{'F1 Score':<25} | {lgb_f1:<18.4f} | {nn_f1:<18.4f} | {nn_f1 - lgb_f1:+10.4f}")
    print(f"{'Brier Score (MSE)':<25} | {lgb_brier:<18.4f} | {nn_brier:<18.4f} | {nn_brier - lgb_brier:+10.4f}")
    print(f"{'Review Flag Rate':<25} | {lgb_review_rate:<18.2%} | {nn_review_rate:<18.2%} | {nn_review_rate - lgb_review_rate:+10.2%}")
    print("-" * 70)

    # 5. Output comparison JSON
    comparison_data = {
        "dataset": "held_out_synthetic_test_set",
        "n_samples": n_samples,
        "n_positive": n_positive,
        "target_definition": "synthetic_missed_payment (generated toy label; positive class 1)",
        "evaluation_protocol": {
            "labels_and_predictions": "test.csv labels; test split used only for final evaluation",
            "lightgbm_threshold_source": "packaged model bundle threshold_from_validation",
            "neural_threshold_source": "backend/artifacts/rakshak_model_metadata.json review_threshold",
            "neural_feature_mapping": "77-column checkpoint contract; test rows mapped from 26-feature test.csv plus neural_sample_payload.json defaults for unavailable fields",
            "caveat": "Neural comparison is not a clean feature-parity experiment: many neural fields are template defaults, not observed per-applicant evidence.",
        },
        "models": {
            "lightgbm_baseline": {
                "name": "LightGBM Gradient Boosted Trees",
                "feature_count": len(FEATURES),
                "threshold": lgb_threshold,
                "roc_auc": round(lgb_auc, 4),
                "accuracy": round(lgb_acc, 4),
                "precision": round(lgb_prec, 4),
                "recall": round(lgb_rec, 4),
                "f1": round(lgb_f1, 4),
                "brier_score": round(lgb_brier, 4),
                "review_rate": round(lgb_review_rate, 4),
                "confusion_matrix": confusion_matrix(y_true, lgb_preds, labels=[0, 1]).tolist(),
                "score_distribution": _distribution(lgb_probs, lgb_threshold),
            },
            "deep_tabular_net": {
                "name": "Rakshak Deep Tabular Neural Network (4-block ResNet)",
                "feature_count": 77,
                "threshold": neural_threshold,
                "roc_auc": round(nn_auc, 4),
                "accuracy": round(nn_acc, 4),
                "precision": round(nn_prec, 4),
                "recall": round(nn_rec, 4),
                "f1": round(nn_f1, 4),
                "brier_score": round(nn_brier, 4),
                "review_rate": round(nn_review_rate, 4),
                "confusion_matrix": confusion_matrix(y_true, neural_preds, labels=[0, 1]).tolist(),
                "score_distribution": _distribution(neural_probs, neural_threshold),
            },
        },
        "regulatory_disclaimer": (
            "All metrics evaluated on synthetic cash-flow labels. "
            "Neither model provides validated default probabilities for automated production lending."
        ),
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(comparison_data, indent=2), encoding="utf-8")
    print(f"\nSaved comparison artifact to {output_file}")
    return comparison_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    run_comparison(parser.parse_args().output)
