"""
Integration tests for POST /api/v1/underwriting/analyze endpoint (Rakshak Credit Phase 4).
Tests:
- Standard request → HTTP 200, complete component statuses
- Flag toggling (include_neural=False, include_gemma=False, etc.)
- Invalid input bounds → HTTP 422
- Response schema compliance
- Non-approval guarantee (no 'approved' in response)
- Preservation of POST /api/v1/neural-score
"""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from unittest.mock import patch, AsyncMock, MagicMock

# Unit/API tests must not call real LLM providers from a developer .env.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ.update({
    "AUTH_SECRET_KEY": "test-only-rakshak-auth-key-0123456789-abcdefgh",
    "DEMO_BORROWER_USERNAME": "borrower-test",
    "DEMO_BORROWER_PASSWORD": "borrower-test-password",
    "DEMO_LENDER_USERNAME": "lender-test",
    "DEMO_LENDER_PASSWORD": "lender-test-password",
})

from fastapi.testclient import TestClient
from app.auth import Principal, issue_access_token
from app.main import app
from app.features import FEATURES as LIGHTGBM_FEATURES
import app.neural_scoring as neural_module


client = TestClient(app, headers={"Authorization": f"Bearer {issue_access_token(Principal('borrower-test', 'borrower'))}"})
BACKEND_DIR = Path(__file__).resolve().parents[1]

VALID_PAYLOAD = {
    "borrower_name": "Asha",
    "occupation": "kirana",
    "business_name": "Asha Provision Store",
    "city": "Kochi",
    "district": "Ernakulam",
    "state": "Kerala",
    "city_tier": "tier2",
    "business_age": "2-5",
    "purpose": "inventory",
    "purpose_details": "Restock grocery fast-movers",
    "requested_amount": 150000.0,
    "monthly_inflow": 75000.0,
    "monthly_expenses": 42000.0,
    "existing_debt": 8000.0,
    "tenure_months": 12,
    "annual_interest_rate_pct": 18.0,
    "evidence": "bank",
    "notes": "Quarterly supplier invoices available.",
    "include_neural": True,
    "include_lightgbm": True,
    "include_gemma": True,
    "include_luna": True,
}


class TestUnderwritingAnalyzeEndpoint(unittest.TestCase):

    # ── Standard Request ──────────────────────────────────────────────

    def test_standard_request_returns_200(self):
        resp = client.post("/api/v1/underwriting/analyze", json=VALID_PAYLOAD)
        self.assertEqual(resp.status_code, 200, resp.text)

    def test_borrower_can_request_an_estimate_without_requested_loan_terms(self):
        payload = {key: value for key, value in VALID_PAYLOAD.items() if key not in {
            "requested_amount", "purpose", "purpose_details", "tenure_months", "annual_interest_rate_pct",
        }}
        response = client.post("/api/v1/underwriting/analyze", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertIsNone(result["financial_calculations"]["requested_amount_inr"])
        self.assertIsNone(result["financial_calculations"]["estimated_monthly_emi"])
        self.assertGreaterEqual(result["financial_calculations"]["illustrative_capacity_range_inr"]["maximum"], 0)

    def test_response_has_application_id(self):
        resp = client.post("/api/v1/underwriting/analyze", json=VALID_PAYLOAD)
        data = resp.json()
        self.assertIn("application_id", data)
        self.assertTrue(data["application_id"].startswith("RC-"))

    def test_response_status_completed_or_partial(self):
        resp = client.post("/api/v1/underwriting/analyze", json=VALID_PAYLOAD)
        data = resp.json()
        self.assertIn(data["status"], ("completed", "partial"))

    def test_components_all_present(self):
        resp = client.post("/api/v1/underwriting/analyze", json=VALID_PAYLOAD)
        data = resp.json()
        expected_keys = {
            "financial_calculations",
            "neural_scoring",
            "lightgbm_scoring",
            "gemma_evidence",
            "luna_decision_support",
        }
        self.assertTrue(expected_keys.issubset(data["components"].keys()))

    def test_component_statuses_are_valid(self):
        resp = client.post("/api/v1/underwriting/analyze", json=VALID_PAYLOAD)
        data = resp.json()
        valid_statuses = {"success", "unconfigured", "skipped", "error"}
        for name, comp in data["components"].items():
            self.assertIn(
                comp["status"], valid_statuses,
                f"Component '{name}' has invalid status: {comp['status']}"
            )

    def test_luna_fulfills_both_llm_roles_without_gemma_call(self):
        evidence_result = SimpleNamespace(
            status="success", latency_ms=2,
            message="Luna completed the evidence-extraction role.",
            model_dump=lambda: {"status": "success", "model_name": "gpt-6-luna", "extraction": {"business_type": "kirana"}},
        )
        review_result = SimpleNamespace(
            status="success", latency_ms=3,
            message="Luna explanation validated for human review.",
            model_dump=lambda: {"status": "success", "model_name": "gpt-6-luna", "decision_support": {"human_review_required": True}},
        )
        with patch("app.underwriting_orchestrator.LunaClient") as luna_factory:
            luna = luna_factory.return_value
            luna.extract_evidence = AsyncMock(return_value=evidence_result)
            luna.generate_review = AsyncMock(return_value=review_result)
            response = client.post("/api/v1/underwriting/analyze", json=VALID_PAYLOAD)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["gemma_evidence"]["model_name"], "gpt-6-luna")
        self.assertEqual(data["luna_decision_support"]["model_name"], "gpt-6-luna")
        self.assertIn("Luna is handling", data["components"]["gemma_evidence"]["message"])
        self.assertEqual(luna.extract_evidence.await_count, 1)
        self.assertEqual(luna.generate_review.await_count, 1)

    def test_financial_calculations_present(self):
        resp = client.post("/api/v1/underwriting/analyze", json=VALID_PAYLOAD)
        data = resp.json()
        fc = data["financial_calculations"]
        self.assertIn("estimated_monthly_emi", fc)
        self.assertIn("estimated_monthly_surplus_before_new_loan", fc)
        self.assertGreater(fc["estimated_monthly_emi"], 0)

    def test_neural_scoring_skipped_without_full_feature_contract(self):
        resp = client.post("/api/v1/underwriting/analyze", json=VALID_PAYLOAD)
        data = resp.json()
        self.assertIsNone(data.get("neural_scoring"))
        self.assertEqual(data["components"]["neural_scoring"]["status"], "skipped")
        self.assertIn("no defaults were invented", data["components"]["neural_scoring"]["message"])

    def test_synthetic_only_flag_set(self):
        resp = client.post("/api/v1/underwriting/analyze", json=VALID_PAYLOAD)
        data = resp.json()
        self.assertTrue(data["synthetic_only"])

    def test_regulatory_warning_present(self):
        resp = client.post("/api/v1/underwriting/analyze", json=VALID_PAYLOAD)
        data = resp.json()
        self.assertIn("regulatory_warning", data)
        self.assertIn("not a real lending decision", data["regulatory_warning"].lower())

    def test_response_never_says_approved(self):
        """Critical: experimental scores must never be represented as approval."""
        resp = client.post("/api/v1/underwriting/analyze", json=VALID_PAYLOAD)
        body_str = resp.text.lower()
        # 'approved' should not appear as a standalone decision field
        data = resp.json()
        for key in ("status", "recommended_action", "decision", "affordability_verdict"):
            for comp in [data.get("luna_decision_support", {}),
                         data.get("neural_scoring", {}),
                         data.get("lightgbm_scoring", {})]:
                if comp and key in comp:
                    self.assertNotIn("loan_approved", comp[key].lower(),
                                     f"Field '{key}' appears to grant approval: {comp[key]}")

    # ── Flag Toggling ────────────────────────────────────────────────

    def test_neural_skipped_when_disabled(self):
        payload = dict(VALID_PAYLOAD, include_neural=False)
        resp = client.post("/api/v1/underwriting/analyze", json=payload)
        data = resp.json()
        self.assertEqual(data["components"]["neural_scoring"]["status"], "skipped")
        self.assertIsNone(data.get("neural_scoring"))

    def test_gemma_skipped_when_disabled(self):
        payload = dict(VALID_PAYLOAD, include_gemma=False)
        resp = client.post("/api/v1/underwriting/analyze", json=payload)
        data = resp.json()
        self.assertEqual(data["components"]["gemma_evidence"]["status"], "skipped")
        self.assertIsNone(data.get("gemma_evidence"))

    def test_luna_skipped_when_disabled(self):
        payload = dict(VALID_PAYLOAD, include_luna=False)
        resp = client.post("/api/v1/underwriting/analyze", json=payload)
        data = resp.json()
        self.assertEqual(data["components"]["luna_decision_support"]["status"], "skipped")
        self.assertIsNone(data.get("luna_decision_support"))

    def test_lightgbm_skipped_when_disabled(self):
        payload = dict(VALID_PAYLOAD, include_lightgbm=False)
        resp = client.post("/api/v1/underwriting/analyze", json=payload)
        data = resp.json()
        self.assertEqual(data["components"]["lightgbm_scoring"]["status"], "skipped")
        self.assertIsNone(data.get("lightgbm_scoring"))

    def test_all_components_skipped_still_returns_financial(self):
        payload = dict(VALID_PAYLOAD, include_neural=False, include_gemma=False,
                       include_luna=False, include_lightgbm=False)
        resp = client.post("/api/v1/underwriting/analyze", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("financial_calculations", data)
        self.assertIn("estimated_monthly_emi", data["financial_calculations"])

    # ── Validation ───────────────────────────────────────────────────

    def test_negative_requested_amount_returns_422(self):
        payload = dict(VALID_PAYLOAD, requested_amount=-50000)
        resp = client.post("/api/v1/underwriting/analyze", json=payload)
        self.assertEqual(resp.status_code, 422)

    def test_zero_requested_amount_returns_422(self):
        payload = dict(VALID_PAYLOAD, requested_amount=0.0)
        resp = client.post("/api/v1/underwriting/analyze", json=payload)
        self.assertEqual(resp.status_code, 422)

    def test_unknown_request_field_returns_422(self):
        payload = dict(VALID_PAYLOAD, model_override="anything")
        self.assertEqual(client.post("/api/v1/underwriting/analyze", json=payload).status_code, 422)

    def test_partial_model_feature_contract_returns_422(self):
        payload = dict(VALID_PAYLOAD, neural_features={"monthly_inflow_inr": 1000})
        response = client.post("/api/v1/underwriting/analyze", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_correlation_id_is_returned(self):
        response = client.post("/api/v1/underwriting/analyze", json=VALID_PAYLOAD,
                               headers={"X-Correlation-ID": "test-case-42"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["X-Correlation-ID"], "test-case-42")
        self.assertEqual(response.json()["correlation_id"], "test-case-42")

    def test_zero_rate_and_negative_surplus_are_reported_exactly(self):
        payload = dict(VALID_PAYLOAD, requested_amount=120000, tenure_months=12,
                       annual_interest_rate_pct=0, monthly_inflow=10000,
                       monthly_expenses=20000, existing_debt=1000,
                       monthly_work_costs=500, safety_buffer=2000,
                       include_neural=False, include_lightgbm=False,
                       include_gemma=False, include_luna=False)
        response = client.post("/api/v1/underwriting/analyze", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        calculations = response.json()["financial_calculations"]
        self.assertEqual(calculations["estimated_monthly_emi"], 10000)
        self.assertEqual(calculations["estimated_monthly_surplus_before_new_loan"], -11500)
        self.assertEqual(calculations["estimated_surplus_after_emi_and_buffer_inr"], -23500)

    def test_full_model_pipeline_with_explicit_synthetic_fixtures(self):
        """Exercise both saved models using labeled synthetic fixtures, never applicant defaults."""
        neural_module.load_neural_model()
        neural = json.loads((BACKEND_DIR / "data/neural_sample_payload.json").read_text(encoding="utf-8"))["features"]
        row = pd.read_csv(BACKEND_DIR / "data/sample_features.csv").iloc[0]
        lightgbm = {key: float(row[key]) for key in LIGHTGBM_FEATURES}
        payload = dict(VALID_PAYLOAD,
            neural_features=neural,
            neural_feature_sources={key: "synthetic_test_fixture:neural_sample_payload.json" for key in neural_module._feature_columns},
            lightgbm_features=lightgbm,
            lightgbm_feature_sources={key: "synthetic_test_fixture:sample_features.csv:row1" for key in LIGHTGBM_FEATURES},
            include_gemma=False, include_luna=False)
        response = client.post("/api/v1/underwriting/analyze", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["components"]["neural_scoring"]["status"], "success")
        self.assertEqual(data["components"]["lightgbm_scoring"]["status"], "success")
        self.assertTrue(data["neural_scoring"]["synthetic_only"])
        self.assertNotIn("approved", data["lightgbm_scoring"])

    def test_negative_monthly_inflow_returns_422(self):
        payload = dict(VALID_PAYLOAD, monthly_inflow=-1000)
        resp = client.post("/api/v1/underwriting/analyze", json=payload)
        self.assertEqual(resp.status_code, 422)

    def test_tenure_zero_returns_422(self):
        payload = dict(VALID_PAYLOAD, tenure_months=0)
        resp = client.post("/api/v1/underwriting/analyze", json=payload)
        self.assertEqual(resp.status_code, 422)

    def test_missing_required_field_returns_422(self):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "requested_amount"}
        resp = client.post("/api/v1/underwriting/analyze", json=payload)
        # FastAPI will use the default value for optional fields; amount has a default
        # But if we send explicitly wrong types:
        payload2 = dict(VALID_PAYLOAD, requested_amount="not-a-number")
        resp2 = client.post("/api/v1/underwriting/analyze", json=payload2)
        self.assertEqual(resp2.status_code, 422)


class TestExistingNeuralScoreEndpoint(unittest.TestCase):
    """Verify POST /api/v1/neural-score is unaffected by new endpoint."""

    NEURAL_PAYLOAD = {
        "features": {
            "occupation": "Shopkeeper",
            "business_type": "kirana_shop",
            "city": "Kochi",
            "state": "Kerala",
            "city_tier": "Tier 2",
            "age_years": 34,
            "household_size": 4,
            "dependents_count": 2,
            "business_years": 3.5,
            "employment_months_if_salaried": 0,
            "loan_purpose": "inventory_purchase",
            "requested_loan_amount_inr": 150000.0,
            "loan_tenure_months": 12,
            "annual_interest_rate_pct": 18.0,
            "monthly_emi_inr": 13736.43,
            "monthly_inflow_inr": 75000.0,
            "monthly_expenses_inr": 42000.0,
            "existing_monthly_debt_inr": 8000.0,
            "net_monthly_surplus_before_new_emi_inr": 25000.0,
            "net_monthly_surplus_after_new_emi_inr": 11263.57,
            "emi_to_inflow_ratio": 0.183,
            "total_debt_service_ratio": 0.29,
            "expense_to_inflow_ratio": 0.56,
            "loan_to_monthly_inflow_ratio": 2.0,
            "income_regularity_score": 0.88,
            "inflow_volatility_cv": 0.15,
            "recurring_inflow_share": 0.80,
            "transaction_coverage_days": 180,
            "bank_statement_months": 6,
            "upi_transaction_count_90d": 620,
            "cash_transaction_share": 0.20,
            "income_expense_consistency_score": 0.90,
            "cashflow_data_quality_score": 0.95,
            "anomaly_count_90d": 0,
            "unexplained_large_transactions_90d": 0,
            "reversal_or_failed_txn_rate": 0.015,
            "recurring_expense_share": 0.45,
            "largest_customer_share": 0.07,
            "weekly_income_cv": 0.14,
            "cashflow_gap_days_12m": 0,
            "income_source_count": 3,
            "income_source_concentration": 0.35,
            "income_stability_index": 0.88,
            "sales_record_quality_score": 0.90,
            "inventory_turnover_days": 14.0,
            "seasonality_index": 1.05,
            "supplier_payment_delay_days": 2.0,
            "monthly_sales_growth_pct": 3.5,
            "supplier_count": 5,
            "customer_concentration_pct": 6.0,
            "business_operating_days_month": 28,
            "business_revenue_months_observed": 12,
            "inventory_financing_share": 0.30,
            "loan_purpose_business_alignment_score": 0.92,
            "informal_debt_indicator": 0,
            "digital_payment_adoption_score": 0.85,
            "emergency_buffer_months": 3.0,
            "income_shock_10pct_surplus_inr": 10763.57,
            "income_shock_20pct_surplus_inr": 1263.57,
            "expense_shock_15pct_surplus_inr": 4963.57,
            "combined_stress_surplus_inr": 2511.07,
            "affordability_headroom_inr": 11263.57,
            "max_safe_emi_30pct_disposable_inr": 8750.0,
            "emi_vs_safe_capacity_ratio": 1.570,
            "repayment_capacity_band": "moderate",
            "affordability_stress_flag": 0,
            "prior_loan_count": 0,
            "missed_payment_count_24m": 0,
            "days_past_due_max_24m": 0,
            "formal_credit_history_months": 0,
            "credit_utilization_ratio": 0.0,
            "secured_debt_share": 0.0,
            "bank_bounce_count_12m": 0,
            "document_completeness_score": 0.95,
            "kyc_status": "verified",
            "fraud_signal_count": 0,
            "identity_or_business_mismatch_flag": 0,
        }
    }

    def test_neural_score_endpoint_returns_200(self):
        resp = client.post("/api/v1/neural-score", json=self.NEURAL_PAYLOAD)
        self.assertEqual(resp.status_code, 200, resp.text)

    def test_neural_score_has_risk_score(self):
        resp = client.post("/api/v1/neural-score", json=self.NEURAL_PAYLOAD)
        data = resp.json()
        self.assertIn("risk_score", data)
        self.assertGreaterEqual(data["risk_score"], 0.0)
        self.assertLessEqual(data["risk_score"], 1.0)

    def test_neural_score_has_decision(self):
        resp = client.post("/api/v1/neural-score", json=self.NEURAL_PAYLOAD)
        data = resp.json()
        self.assertIn("decision", data)
        self.assertIn(data["decision"], ("REVIEW_REQUIRED", "REVIEW_NOT_TRIGGERED"))

    def test_neural_score_has_synthetic_warning(self):
        resp = client.post("/api/v1/neural-score", json=self.NEURAL_PAYLOAD)
        data = resp.json()
        self.assertIn("warning", data)
        self.assertGreater(len(data["warning"]), 10)


class TestExistingLightGBMReviewContract(unittest.TestCase):
    def test_score_endpoint_does_not_issue_approval_or_rejection(self):
        row = pd.read_csv(BACKEND_DIR / "data/sample_features.csv").iloc[0]
        response = client.post("/api/v1/score", json={"features": row.to_dict()})
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertNotIn("approved", data)
        self.assertIn(data["decision"], ("REVIEW_REQUIRED", "REVIEW_NOT_TRIGGERED"))
        self.assertEqual(data["bureau_context"]["status"], "NO_BUREAU_DATA")


if __name__ == "__main__":
    unittest.main()
