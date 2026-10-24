"""
Step 4D & 4F — Edge Case & Regression Test Suite for Rakshak Neural Scoring
Automates checks for:
- All 77 required features
- Preprocessing and unpickling compatibility
- Missing features & None values (HTTP 422)
- Invalid data types / non-numeric strings in numeric features (HTTP 422)
- Malformed JSON payloads (HTTP 422)
- Unseen / novel categorical values (gracefully handled)
- Output structure (risk_score, threshold, review_required, decision, warning)
- Threshold behavior (probability >= 0.28 triggers REVIEW_REQUIRED)
- Missing model artifacts (HTTP 503)
"""
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

# Prevent backend imports from loading local LLM credentials during tests.
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
from app.neural_scoring import load_neural_model, neural_model_score
import app.neural_scoring as ns

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "neural_sample_payload.json"


class TestNeuralScoringEdgeCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app, headers={"Authorization": f"Bearer {issue_access_token(Principal('borrower-test', 'borrower'))}"})
        with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
            cls.valid_payload = json.load(f)
        load_neural_model()

    def test_01_valid_77_feature_inference_success(self):
        """Test standard 77-feature inference returns HTTP 200 with complete structure."""
        response = self.client.post("/api/v1/neural-score", json=self.valid_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("risk_score", data)
        self.assertIn("threshold", data)
        self.assertIn("review_required", data)
        self.assertIn("decision", data)
        self.assertIn("warning", data)
        self.assertIn("synthetic_only", data)
        
        self.assertEqual(data["threshold"], 0.28)
        self.assertIsInstance(data["risk_score"], float)
        self.assertIsInstance(data["review_required"], bool)
        self.assertEqual(data["decision"], "REVIEW_REQUIRED" if data["risk_score"] >= 0.28 else "REVIEW_NOT_TRIGGERED")

    def test_02_empty_features_returns_422(self):
        """Passing an empty features dict triggers HTTP 422 with missing feature list."""
        response = self.client.post("/api/v1/neural-score", json={"features": {}})
        self.assertEqual(response.status_code, 422)
        self.assertIn("Missing neural model features", response.json()["detail"])

    def test_03_missing_single_feature_returns_422(self):
        """Omitting even a single feature of the 77 triggers HTTP 422 naming that feature."""
        incomplete = {
            "features": {
                k: v for k, v in self.valid_payload["features"].items()
                if k != "monthly_inflow_inr"
            }
        }
        response = self.client.post("/api/v1/neural-score", json=incomplete)
        self.assertEqual(response.status_code, 422)
        self.assertIn("monthly_inflow_inr", response.json()["detail"])

    def test_04_none_feature_value_returns_422(self):
        """Passing None for any required feature triggers HTTP 422."""
        with_none = {
            "features": {
                **self.valid_payload["features"],
                "occupation": None,
            }
        }
        response = self.client.post("/api/v1/neural-score", json=with_none)
        self.assertEqual(response.status_code, 422)
        self.assertIn("Missing neural model features: occupation", response.json()["detail"])

    def test_05_invalid_numeric_type_returns_422(self):
        """Non-numeric string in a numeric column triggers HTTP 422."""
        invalid_type = {
            "features": {
                **self.valid_payload["features"],
                "monthly_inflow_inr": "not_a_valid_number",
            }
        }
        response = self.client.post("/api/v1/neural-score", json=invalid_type)
        self.assertEqual(response.status_code, 422)

    def test_06_malformed_json_body_returns_422(self):
        """Invalid schema (missing 'features' key) returns HTTP 422."""
        response = self.client.post("/api/v1/neural-score", json={"unexpected_key": 123})
        self.assertEqual(response.status_code, 422)

    def test_07_unknown_categorical_handled_safely(self):
        """Unseen categorical values (e.g., unknown occupation or city) are handled safely."""
        novel_cat = {
            "features": {
                **self.valid_payload["features"],
                "occupation": "unknown_future_gig_title",
                "city": "UnknownCityOuterSpace",
            }
        }
        response = self.client.post("/api/v1/neural-score", json=novel_cat)
        self.assertEqual(response.status_code, 200)
        self.assertIn("risk_score", response.json())

    def test_08_threshold_behavior_high_risk(self):
        """High-risk cashflow profile triggers review_required = True and score >= 0.28."""
        high_risk = {
            "features": {
                **self.valid_payload["features"],
                "repayment_capacity_band": "stressed",
                "fraud_signal_count": 4,
                "bank_bounce_count_12m": 12,
                "missed_payment_count_24m": 6,
                "days_past_due_max_24m": 90,
                "credit_utilization_ratio": 0.98,
                "monthly_expenses_inr": 48000.0,
                "existing_monthly_debt_inr": 20000.0,
                "net_monthly_surplus_before_new_emi_inr": -19500.0,
                "net_monthly_surplus_after_new_emi_inr": -23625.0,
                "affordability_stress_flag": 1,
            }
        }
        response = self.client.post("/api/v1/neural-score", json=high_risk)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(data["risk_score"], data["threshold"])
        self.assertTrue(data["review_required"])
        self.assertEqual(data["decision"], "REVIEW_REQUIRED")

    def test_09_missing_model_artifact_returns_503(self):
        """When the torch model artifact is missing, HTTP 503 is returned."""
        with patch("app.neural_scoring.MODEL_PATH", Path("non_existent_model_file.pt")):
            saved_model = ns._model
            try:
                ns._model = None  # Reset singleton to test load failure
                response = self.client.post("/api/v1/neural-score", json=self.valid_payload)
                self.assertEqual(response.status_code, 503)
                self.assertIn("Missing model artifact", response.json()["detail"])
            finally:
                ns._model = saved_model

    def test_10_missing_preprocessor_artifact_returns_503(self):
        """When the preprocessor joblib artifact is missing, HTTP 503 is returned."""
        with patch("app.neural_scoring.PREPROCESSOR_PATH", Path("non_existent_preprocessor.joblib")):
            saved_model = ns._model
            try:
                ns._model = None
                response = self.client.post("/api/v1/neural-score", json=self.valid_payload)
                self.assertEqual(response.status_code, 503)
                self.assertIn("Missing model artifact", response.json()["detail"])
            finally:
                ns._model = saved_model

    def test_11_missing_metadata_artifact_returns_503(self):
        """When the metadata json artifact is missing, HTTP 503 is returned."""
        with patch("app.neural_scoring.METADATA_PATH", Path("non_existent_metadata.json")):
            saved_model = ns._model
            try:
                ns._model = None
                response = self.client.post("/api/v1/neural-score", json=self.valid_payload)
                self.assertEqual(response.status_code, 503)
                self.assertIn("Missing model artifact", response.json()["detail"])
            finally:
                ns._model = saved_model

    def test_12_features_wrong_type_non_dict_returns_422(self):
        """Passing a non-dict for 'features' (e.g. list or string) triggers HTTP 422."""
        response = self.client.post("/api/v1/neural-score", json={"features": "not_a_dictionary"})
        self.assertEqual(response.status_code, 422)

        response2 = self.client.post("/api/v1/neural-score", json={"features": [1, 2, 3]})
        self.assertEqual(response2.status_code, 422)

    def test_13_raw_invalid_json_returns_422(self):
        """Passing raw unparseable JSON triggers HTTP 422 from FastAPI."""
        response = self.client.post(
            "/api/v1/neural-score",
            content=b"{ invalid_json: true, ",
            headers={"Content-Type": "application/json"}
        )
        self.assertEqual(response.status_code, 422)

    def test_14_inference_runtime_failure_returns_500(self):
        """Simulate unexpected internal neural network exception returning HTTP 500."""
        with patch.object(ns._model, "forward", side_effect=RuntimeError("Simulated neural inference execution error")):
            response = self.client.post("/api/v1/neural-score", json=self.valid_payload)
            self.assertEqual(response.status_code, 500)
            self.assertIn("Neural scoring failed. Check backend logs.", response.json()["detail"])

    def test_15_dimension_mismatch_returns_422(self):
        """Simulate preprocessor output dimension mismatch raising ValueError and HTTP 422."""
        with patch.object(ns._preprocessor, "transform", return_value=np.zeros((1, 5))):
            response = self.client.post("/api/v1/neural-score", json=self.valid_payload)
            self.assertEqual(response.status_code, 422)
            self.assertIn("dimensions do not match", response.json()["detail"])

    def test_16_unknown_feature_returns_422(self):
        payload = {"features": {**self.valid_payload["features"], "invented_feature": 1}}
        response = self.client.post("/api/v1/neural-score", json=payload)
        self.assertEqual(response.status_code, 422)
        self.assertIn("Unknown neural model features", response.json()["detail"])

    def test_17_non_finite_numeric_feature_returns_422(self):
        features = dict(self.valid_payload["features"])
        features["monthly_inflow_inr"] = float("nan")
        response = self.client.post(
            "/api/v1/neural-score",
            content=json.dumps({"features": features}, allow_nan=True),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("must be finite", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()

