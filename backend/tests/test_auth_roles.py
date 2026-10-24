"""Role-based demo sign-in and lender-only route checks."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cryptography.fernet import Fernet

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ.update({
    "AUTH_SECRET_KEY": "test-only-rakshak-auth-key-0123456789-abcdefgh",
    "DEMO_BORROWER_USERNAME": "borrower-test",
    "DEMO_BORROWER_PASSWORD": "borrower-test-password",
    "DEMO_LENDER_USERNAME": "lender-test",
    "DEMO_LENDER_PASSWORD": "lender-test-password",
})

from fastapi.testclient import TestClient
from app import database
from app.main import app


class TestRoleBasedDemoSignIn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.db_patch = patch.object(database, "DB_PATH", Path(cls.temp_dir.name) / "auth-test.db")
        cls.db_patch.start()
        cls.key_patch = patch.dict(os.environ, {"DOCUMENT_ENCRYPTION_KEY": Fernet.generate_key().decode()})
        cls.key_patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.key_patch.stop()
        cls.db_patch.stop()
        cls.temp_dir.cleanup()

    def test_each_demo_role_can_sign_in_and_get_role_specific_cookie(self):
        with TestClient(app) as client:
            for username, password, role in (
                ("borrower-test", "borrower-test-password", "borrower"),
                ("lender-test", "lender-test-password", "lender"),
            ):
                response = client.post("/api/v1/auth/login", json={"username": username, "password": password})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["user"]["role"], role)
                self.assertNotIn("access_token", response.json())
                self.assertIn("httponly", response.headers["set-cookie"].lower())
                self.assertEqual(client.get("/api/v1/auth/me").json()["role"], role)
                logged_out = client.post("/api/v1/auth/logout")
                self.assertEqual(logged_out.status_code, 204, logged_out.text)
                self.assertIn("max-age=0", logged_out.headers.get("set-cookie", "").lower())
                self.assertEqual(client.get("/api/v1/auth/me").status_code, 401)

    def test_wrong_password_is_rejected(self):
        with TestClient(app) as client:
            response = client.post("/api/v1/auth/login", json={"username": "lender-test", "password": "wrong"})
        self.assertEqual(response.status_code, 401)

    def test_lender_queue_requires_lender_role_and_authentication(self):
        with TestClient(app) as client:
            unauthenticated = client.get("/api/v1/lender/applications")
            self.assertEqual(unauthenticated.status_code, 401)
            borrower = client.post("/api/v1/auth/login", json={"username": "borrower-test", "password": "borrower-test-password"})
            borrower_response = client.get("/api/v1/lender/applications")
            self.assertEqual(borrower_response.status_code, 403)
            client.post("/api/v1/auth/logout")
            lender = client.post("/api/v1/auth/login", json={"username": "lender-test", "password": "lender-test-password"})
            self.assertEqual(lender.status_code, 200)
            lender_response = client.get("/api/v1/lender/applications")
            self.assertEqual(lender_response.status_code, 200, lender_response.text)
            self.assertIn("items", lender_response.json())

    def test_borrower_upload_and_analysis_routes_require_authentication(self):
        with TestClient(app) as client:
            response = client.post("/api/v1/underwriting/applications", json={
                "occupation": "not_employed", "requested_amount": 1000,
                "monthly_inflow": 0, "monthly_expenses": 0, "existing_debt": 0,
                "tenure_months": 1, "annual_interest_rate_pct": 0,
            })
        self.assertEqual(response.status_code, 401)


if __name__ == "__main__":
    unittest.main()
