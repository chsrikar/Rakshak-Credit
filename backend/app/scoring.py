"""
Scoring engine — loads the trained LightGBM model and computes:
  - Illustrative risk probability
  - Rakshak credit score (0-900 scale)
  - Risk tier (Low / Medium / High)
  - Per-applicant SHAP explanations (top contributing features)
  - Safe loan limit estimate
  - Plain-language improvement tips

All outputs are synthetic-only. Not for real lending decisions.
"""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from .features import FEATURES

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "best_model.joblib"

# Lazy-loaded globals
_bundle = None
_shap_explainer = None

# ── Human-readable feature labels ────────────────────────────────────
FEATURE_LABELS = {
    "avg_monthly_income": "Average monthly income",
    "median_monthly_income": "Median monthly income",
    "income_cv": "Income volatility",
    "positive_income_months_ratio": "Months with positive income",
    "income_trend": "Income trend (growth)",
    "avg_monthly_expenses": "Average monthly expenses",
    "essential_expense_ratio": "Essential expense share",
    "savings_ratio": "Savings ratio",
    "avg_closing_balance": "Average closing balance",
    "low_balance_days_ratio": "Days with low balance",
    "net_cash_flow_6m": "Net cash flow (6 months)",
    "negative_cash_flow_months_ratio": "Months with negative cash flow",
    "income_source_concentration": "Income source concentration",
    "transaction_regularity": "Transaction regularity",
    "cash_flow_coverage_ratio": "Cash flow coverage",
    "occupation_gig_worker": "Gig worker indicator",
    "requested_loan_amount_inr": "Requested loan amount",
    "requested_tenure_months": "Requested tenure",
    "self_reported_monthly_essential_expenses_inr": "Self-reported expenses",
    "self_reported_existing_monthly_debt_inr": "Self-reported existing debt",
    "months_in_occupation": "Months in occupation",
    "workdays_per_month": "Work days per month",
    "monthly_work_costs_inr": "Monthly work costs",
    "occupation_numeric_1": "Occupation factor 1",
    "occupation_numeric_2": "Occupation factor 2",
    "context_data_completeness": "Data completeness",
}

# ── Improvement tip templates (keyed by feature direction) ───────────
TIP_TEMPLATES = {
    "income_cv": {
        "high_risk": "Your income varies a lot month to month. Try to maintain steadier earnings — even small regular deposits help.",
        "direction": "lower_is_better",
    },
    "low_balance_days_ratio": {
        "high_risk": "Your account balance drops very low on many days. Keeping a small buffer (even ₹2,000) consistently improves your profile.",
        "direction": "lower_is_better",
    },
    "savings_ratio": {
        "high_risk": "You're spending most of what you earn. Setting aside even 10% monthly signals financial discipline.",
        "direction": "higher_is_better",
    },
    "transaction_regularity": {
        "high_risk": "Irregular transactions make it harder to predict your cash flow. Transacting regularly (UPI, deposits) helps build a pattern.",
        "direction": "higher_is_better",
    },
    "negative_cash_flow_months_ratio": {
        "high_risk": "You've had months where outflows exceeded income. Reducing big one-time expenses or spacing them out helps.",
        "direction": "lower_is_better",
    },
    "essential_expense_ratio": {
        "high_risk": "A large share of your income goes to essentials. Reducing fixed costs or increasing income improves your margin.",
        "direction": "lower_is_better",
    },
    "avg_closing_balance": {
        "high_risk": "Your average balance is low. Even depositing a small fixed amount weekly can raise this score driver.",
        "direction": "higher_is_better",
    },
    "income_trend": {
        "high_risk": "Your income has been trending down recently. Lenders look for stable or growing earnings.",
        "direction": "higher_is_better",
    },
    "monthly_work_costs_inr": {
        "high_risk": "High work costs eat into your net income. Optimizing fuel, maintenance, or supply costs helps your profile.",
        "direction": "lower_is_better",
    },
}


def _load_bundle():
    """Load model bundle once and cache it."""
    global _bundle
    if _bundle is not None:
        return _bundle

    if not MODEL_PATH.exists():
        _bundle = None
        return None

    import joblib
    _bundle = joblib.load(MODEL_PATH)
    return _bundle


def _get_shap_explainer(model):
    """Create and cache a SHAP TreeExplainer."""
    global _shap_explainer
    if _shap_explainer is not None:
        return _shap_explainer
    try:
        import shap
        _shap_explainer = shap.TreeExplainer(model)
        return _shap_explainer
    except Exception:
        return None


def _prob_to_rakshak_score(prob: float) -> int:
    """
    Convert miss-payment probability (0=safe, 1=risky)
    to a Rakshak score (300-900 scale, higher = better).
    """
    # Invert: low risk → high score
    score = 300 + int((1 - prob) * 600)
    return max(300, min(900, score))


def _risk_tier(score: int) -> str:
    if score >= 700:
        return "Low Risk"
    elif score >= 500:
        return "Medium Risk"
    else:
        return "High Risk"


def _tier_color(tier: str) -> str:
    return {
        "Low Risk": "#10b981",
        "Medium Risk": "#f59e0b",
        "High Risk": "#ef4444",
    }.get(tier, "#6b7280")


def _generate_tips(features: dict, shap_values: dict) -> list:
    """Generate improvement tips based on SHAP contributions."""
    tips = []
    # Sort features by absolute SHAP contribution (biggest risk drivers first)
    sorted_features = sorted(
        shap_values.items(), key=lambda x: abs(x[1]), reverse=True
    )
    for feat, shap_val in sorted_features[:5]:
        if feat in TIP_TEMPLATES and shap_val > 0.01:
            # Positive SHAP = increases risk probability
            tips.append({
                "feature": feat,
                "feature_label": FEATURE_LABELS.get(feat, feat),
                "impact": "increases_risk",
                "shap_contribution": round(shap_val, 4),
                "tip": TIP_TEMPLATES[feat]["high_risk"],
            })
    return tips[:3]  # Top 3 actionable tips


def model_score(features: dict) -> dict:
    """Score a single applicant and return full explanation."""
    if not isinstance(features, dict):
        raise ValueError("Features must be an object.")
    missing = [f for f in FEATURES if f not in features]
    if missing:
        raise ValueError(
            "Missing required feature(s): " + ", ".join(missing)
        )

    unknown = sorted(set(features) - set(FEATURES))
    if unknown:
        raise ValueError("Unknown feature(s): " + ", ".join(unknown))
    row = {}
    for name in FEATURES:
        try:
            value = float(features[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature '{name}' must be numeric.") from exc
        if not math.isfinite(value):
            raise ValueError(f"Feature '{name}' must be finite.")
        row[name] = value
    frame = pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)

    bundle = _load_bundle()

    if bundle is not None:
        model = bundle["model"]
        prob = float(model.predict_proba(frame)[0, 1])
        model_name = bundle.get("model_name", "LightGBM")
        source = "trained_synthetic_model"
        threshold = bundle.get("threshold_from_validation", 0.30)

        # SHAP explanations
        shap_contributions = {}
        top_reasons = []
        explainer = _get_shap_explainer(model)
        if explainer is not None:
            try:
                sv = explainer.shap_values(frame)
                if isinstance(sv, list):
                    sv = sv[1]  # class-1 SHAP values
                sv = np.asarray(sv)
                if sv.ndim == 3:
                    sv = sv[:, :, 1]
                shap_row = sv[0]
                shap_contributions = {
                    f: round(float(shap_row[i]), 4)
                    for i, f in enumerate(FEATURES)
                }
                # Top positive/negative drivers
                sorted_shap = sorted(
                    shap_contributions.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )
                for feat, val in sorted_shap[:5]:
                    label = FEATURE_LABELS.get(feat, feat)
                    if val > 0.01:
                        top_reasons.append(
                            f"[+] {label} raises risk (SHAP: +{val:.3f})"
                        )
                    elif val < -0.01:
                        top_reasons.append(
                            f"[-] {label} lowers risk (SHAP: {val:.3f})"
                        )
            except Exception:
                pass  # Graceful fallback
    else:
        # Heuristic fallback if model not packaged yet
        income = max(row["avg_monthly_income"], 1)
        burden = min(2, max(0, row["avg_monthly_expenses"] / income))
        volatility = min(1.5, max(0, row["income_cv"]))
        low_buffer = 1 if row["avg_closing_balance"] < 3000 else 0
        logit = (
            -2.3
            + 1.65 * volatility
            + 1.65 * burden
            + 0.65 * max(0, row["negative_cash_flow_months_ratio"])
            + 0.45 * low_buffer
        )
        prob = float(1 / (1 + math.exp(-logit)))
        model_name = "transparent demo heuristic"
        source = "heuristic_not_validated"
        threshold = 0.30
        shap_contributions = {}
        top_reasons = [
            "Heuristic model — no SHAP explanations available.",
            "Package the trained model for full explanations.",
        ]

    rakshak_score = _prob_to_rakshak_score(prob)
    tier = _risk_tier(rakshak_score)
    tips = _generate_tips(row, shap_contributions) if shap_contributions else []

    # The demo cannot turn missing bureau history into a rejection.
    bureau_context = {
        "status": "NO_BUREAU_DATA",
        "reason": "No bureau history was supplied; no bureau decision is produced.",
        "score": None,
    }

    return {
        "rakshak_score": rakshak_score,
        "risk_probability": round(prob, 4),
        "score_percent": round(prob * 100, 1),
        "risk_tier": tier,
        "tier_color": _tier_color(tier),
        "threshold_used": threshold,
        "review_required": prob >= threshold,
        "decision": "REVIEW_REQUIRED" if prob >= threshold else "REVIEW_NOT_TRIGGERED",
        "model_name": model_name,
        "score_source": source,
        "shap_contributions": shap_contributions,
        "top_reasons": top_reasons,
        "improvement_tips": tips,
        "bureau_context": bureau_context,
        "synthetic_only": True,
        "warning": (
            "Not a real credit score. Not validated on real repayment outcomes; "
            "do not use for lending decisions."
        ),
    }


def affordability_estimate(
    monthly_income,
    essential_expenses,
    existing_monthly_debt,
    monthly_work_costs,
    safety_buffer,
    repayment_share,
    monthly_interest_rate,
    tenure_months,
):
    available = max(
        0.0,
        monthly_income
        - essential_expenses
        - existing_monthly_debt
        - monthly_work_costs
        - safety_buffer,
    )
    payment = available * repayment_share
    rate = monthly_interest_rate
    if rate > 0:
        principal = payment * (1 - (1 + rate) ** (-tenure_months)) / rate
    else:
        principal = payment * tenure_months

    return {
        "estimated_monthly_surplus_before_buffer": round(
            max(
                0,
                monthly_income
                - essential_expenses
                - existing_monthly_debt
                - monthly_work_costs,
            ),
            2,
        ),
        "safety_buffer_inr": round(safety_buffer, 2),
        "estimated_available_amount_inr": round(available, 2),
        "illustrative_monthly_payment_inr": round(payment, 2),
        "illustrative_principal_inr": round(max(0, principal), 2),
        "tenure_months": tenure_months,
        "monthly_interest_rate_assumption": rate,
        "repayment_share_assumption": repayment_share,
        "demo_only": True,
        "warning": (
            "Illustrative arithmetic only; not an offer, "
            "eligibility decision, or financial advice."
        ),
    }
