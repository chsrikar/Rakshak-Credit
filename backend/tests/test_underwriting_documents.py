"""Local encrypted applicant-draft and document-upload API tests."""
import os
import sqlite3
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
from app.auth import Principal, issue_access_token
from app.main import app


PAYLOAD = {
    "borrower_name": "Synthetic Upload Test",
    "occupation": "not_employed",
    "business_name": "",
    "city": "Kochi",
    "district": "Ernakulam",
    "state": "Kerala",
    "purpose": "personal",
    "purpose_details": "Synthetic test only",
    "requested_amount": 50000,
    "monthly_inflow": 30000,
    "monthly_expenses": 18000,
    "existing_debt": 0,
    "tenure_months": 12,
    "annual_interest_rate_pct": 18,
}
PDF_FIXTURE = b"%PDF-1.4\nSynthetic test file; no identity or financial data.\n%%EOF"


class TestUnderwritingDocumentAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls.temp_dir.name) / "test.db"
        cls.db_patch = patch.object(database, "DB_PATH", cls.db_path)
        cls.db_patch.start()
        cls.key_patch = patch.dict(os.environ, {"DOCUMENT_ENCRYPTION_KEY": Fernet.generate_key().decode()})
        cls.key_patch.start()
        cls.client = TestClient(app, headers={
            "Authorization": f"Bearer {issue_access_token(Principal('borrower-test', 'borrower'))}"
        })
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.key_patch.stop()
        cls.db_patch.stop()
        cls.temp_dir.cleanup()

    def create_application(self):
        response = self.client.post("/api/v1/underwriting/applications", json=PAYLOAD)
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["application_id"]

    def upload_required_documents(self, application_id):
        for category in ("identity_pan", "bank_statement", "tax_return"):
            response = self.client.post(
                f"/api/v1/underwriting/applications/{application_id}/documents",
                data={"category": category},
                files={"file": (f"{category}.pdf", PDF_FIXTURE, "application/pdf")},
            )
            self.assertEqual(response.status_code, 201, response.text)

    def test_borrower_profile_can_be_saved_without_loan_request_details(self):
        profile = {key: value for key, value in PAYLOAD.items() if key not in {
            "requested_amount", "purpose", "purpose_details", "tenure_months", "annual_interest_rate_pct",
        }}
        response = self.client.post("/api/v1/underwriting/applications", json=profile)
        self.assertEqual(response.status_code, 201, response.text)
        self.assertTrue(response.json()["application_id"].startswith("RC-"))

    def test_application_profile_is_encrypted_in_database(self):
        application_id = self.create_application()
        conn = sqlite3.connect(self.db_path)
        try:
            encrypted = conn.execute(
                "SELECT payload_encrypted FROM underwriting_applications WHERE id=?",
                (application_id,),
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertNotIn(b"Synthetic Upload Test", encrypted)
        self.assertGreater(len(encrypted), 64)

    def test_supported_file_upload_is_encrypted_and_lists_metadata_only(self):
        application_id = self.create_application()
        response = self.client.post(
            f"/api/v1/underwriting/applications/{application_id}/documents",
            data={"category": "identity_pan"},
            files={"file": ("private-pan.pdf", PDF_FIXTURE, "application/pdf")},
        )
        self.assertEqual(response.status_code, 201, response.text)
        metadata = response.json()
        self.assertEqual(metadata["category"], "identity_pan")
        self.assertNotIn("encrypted_content", metadata)
        self.assertNotIn("private-pan", metadata["stored_filename"])

        conn = sqlite3.connect(self.db_path)
        try:
            encrypted = conn.execute(
                "SELECT encrypted_content FROM underwriting_documents WHERE id=?",
                (metadata["id"],),
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertNotIn(PDF_FIXTURE, encrypted)
        self.assertGreater(len(encrypted), len(PDF_FIXTURE))

        listed = self.client.get(f"/api/v1/underwriting/applications/{application_id}/documents")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(len(listed.json()["items"]), 1)
        self.assertNotIn("encrypted_content", listed.json()["items"][0])
        removed = self.client.delete(
            f"/api/v1/underwriting/applications/{application_id}/documents/{metadata['id']}"
        )
        self.assertEqual(removed.status_code, 200)
        self.assertEqual(self.client.get(f"/api/v1/underwriting/applications/{application_id}/documents").json()["items"], [])

    def test_other_supporting_document_can_be_uploaded(self):
        application_id = self.create_application()
        response = self.client.post(
            f"/api/v1/underwriting/applications/{application_id}/documents",
            data={"category": "other"},
            files={"file": ("supporting-document.pdf", PDF_FIXTURE, "application/pdf")},
        )
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["category"], "other")

    def test_upload_rejects_unknown_category_and_invalid_file(self):
        application_id = self.create_application()
        path = f"/api/v1/underwriting/applications/{application_id}/documents"
        bad_category = self.client.post(path, data={"category": "passport"}, files={"file": ("x.pdf", PDF_FIXTURE)})
        self.assertEqual(bad_category.status_code, 422)
        bad_file = self.client.post(path, data={"category": "other"}, files={"file": ("x.pdf", b"not a PDF", "application/pdf")})
        self.assertEqual(bad_file.status_code, 415)

    def test_upload_rejects_file_over_limit(self):
        application_id = self.create_application()
        too_large = b"%PDF-1.4\n" + (b"x" * (12 * 1024 * 1024 + 1))
        response = self.client.post(
            f"/api/v1/underwriting/applications/{application_id}/documents",
            data={"category": "other"},
            files={"file": ("large.pdf", too_large, "application/pdf")},
        )
        self.assertEqual(response.status_code, 413)

    def test_missing_encryption_key_fails_closed(self):
        application_id = self.create_application()
        with patch.dict(os.environ, {"DOCUMENT_ENCRYPTION_KEY": ""}):
            response = self.client.post(
                f"/api/v1/underwriting/applications/{application_id}/documents",
                data={"category": "other"},
                files={"file": ("test.pdf", PDF_FIXTURE, "application/pdf")},
            )
        self.assertEqual(response.status_code, 503)

    def test_documents_require_existing_application(self):
        response = self.client.get("/api/v1/underwriting/applications/RC-NOT-FOUND/documents")
        self.assertEqual(response.status_code, 404)

    def test_lender_can_download_uploaded_document(self):
        application_id = self.create_application()
        uploaded = self.client.post(
            f"/api/v1/underwriting/applications/{application_id}/documents",
            data={"category": "other"},
            files={"file": ("demo.pdf", PDF_FIXTURE, "application/pdf")},
        )
        self.assertEqual(uploaded.status_code, 201, uploaded.text)
        lender = TestClient(app)
        lender.post("/api/v1/auth/login", json={"username": "lender-test", "password": "lender-test-password"})
        response = lender.get(
            f"/api/v1/underwriting/applications/{application_id}/documents/{uploaded.json()['id']}/download"
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.content, PDF_FIXTURE)
        self.assertIn("attachment", response.headers["content-disposition"])
        lender.close()

    def test_uploaded_synthetic_pdf_text_can_be_retrieved_without_exposing_identifiers(self):
        application_id = self.create_application()
        sample_path = Path(__file__).resolve().parents[2] / "output" / "pdf" / "synthetic-demo-bank-statement.pdf"
        sample = sample_path.read_bytes()
        uploaded = self.client.post(
            f"/api/v1/underwriting/applications/{application_id}/documents",
            data={"category": "bank_statement"},
            files={"file": (sample_path.name, sample, "application/pdf")},
        )
        self.assertEqual(uploaded.status_code, 201, uploaded.text)
        response = self.client.get(
            f"/api/v1/underwriting/applications/{application_id}/documents/{uploaded.json()['id']}/text"
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "success")
        self.assertIn("NOT A REAL DOCUMENT", response.json()["text"])
        self.assertIn("[REDACTED 12-DIGIT ID]", response.json()["text"])

    def test_analysis_sends_extracted_document_text_to_luna(self):
        application_id = self.create_application()
        self.upload_required_documents(application_id)
        extracted = [{
            "document_id": "synthetic-statement-1", "category": "bank_statement",
            "content_type": "application/pdf", "status": "success",
            "text": "DEMO CREDIT INR 12,000", "message": "Extracted sample text.",
        }]
        evidence_result = type("Result", (), {
            "status": "success", "latency_ms": 1, "message": "Evidence extracted.",
            "model_dump": lambda self: {"status": "success", "model_name": "test-luna", "extraction": {}},
        })()
        review_result = type("Result", (), {
            "status": "success", "latency_ms": 1, "message": "Review generated.",
            "model_dump": lambda self: {"status": "success", "model_name": "test-luna", "decision_support": {}},
        })()
        from unittest.mock import AsyncMock
        with patch("app.underwriting_orchestrator.extract_application_documents", return_value=extracted), \
             patch("app.underwriting_orchestrator.LunaClient") as client_factory:
            luna = client_factory.return_value
            luna.extract_evidence = AsyncMock(return_value=evidence_result)
            luna.generate_review = AsyncMock(return_value=review_result)
            response = self.client.post("/api/v1/underwriting/analyze", json={
                **PAYLOAD, "application_id": application_id,
                "include_neural": False, "include_lightgbm": False,
            })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["components"]["document_extraction"]["status"], "success")
        sent_documents = luna.extract_evidence.await_args.args[0]["uploaded_documents"]
        self.assertEqual(sent_documents[0]["text"], "DEMO CREDIT INR 12,000")
        self.assertEqual(sent_documents[0]["document_id"], "synthetic-statement-1")

    def test_borrower_cannot_access_another_owners_application(self):
        application_id = "RC-OTHER-OWNER-TEST"
        database.save_underwriting_application(application_id, PAYLOAD, owner_id="different-borrower")
        response = self.client.get(f"/api/v1/underwriting/applications/{application_id}")
        self.assertEqual(response.status_code, 404)

    def test_only_lender_can_update_persistent_human_review_status(self):
        application_id = self.create_application()
        borrower_response = self.client.put(
            f"/api/v1/lender/applications/{application_id}/review",
            json={"status": "in_review"},
        )
        self.assertEqual(borrower_response.status_code, 403)
        lender = TestClient(app, headers={
            "Authorization": f"Bearer {issue_access_token(Principal('lender-test', 'lender'))}"
        })
        try:
            updated = lender.put(
                f"/api/v1/lender/applications/{application_id}/review",
                json={"status": "in_review"},
            )
            self.assertEqual(updated.status_code, 200, updated.text)
            queue = lender.get("/api/v1/lender/applications")
            record = next(item for item in queue.json()["items"] if item["application_id"] == application_id)
            self.assertEqual(record["review_status"], "in_review")
            self.assertEqual(record["reviewed_by"], "lender-test")
        finally:
            lender.close()

    def test_analysis_result_is_encrypted_and_linked_to_saved_application(self):
        application_id = self.create_application()
        self.upload_required_documents(application_id)
        payload = {
            **PAYLOAD,
            "application_id": application_id,
            "include_neural": False,
            "include_lightgbm": False,
            "include_gemma": False,
            "include_luna": False,
        }
        response = self.client.post("/api/v1/underwriting/analyze", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["application_id"], application_id)
        self.assertEqual(response.json()["components"]["database_persistence"]["status"], "success")
        conn = sqlite3.connect(self.db_path)
        try:
            encrypted = conn.execute(
                "SELECT response_encrypted FROM underwriting_analyses WHERE application_id=?",
                (application_id,),
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertNotIn(application_id.encode(), encrypted)

    def test_estimate_requires_identity_bank_statement_and_tax_return(self):
        application_id = self.create_application()
        response = self.client.post("/api/v1/underwriting/analyze", json={
            **PAYLOAD, "application_id": application_id,
            "include_neural": False, "include_lightgbm": False,
            "include_gemma": False, "include_luna": False,
        })
        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(set(response.json()["detail"]["missing_document_categories"]), {
            "identity_pan", "bank_statement", "tax_return",
        })


if __name__ == "__main__":
    unittest.main()
