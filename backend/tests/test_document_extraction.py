import unittest
from unittest.mock import patch

from app.document_extraction import cap_document_texts, extract_document_text, redact_sensitive_identifiers


class FakePage:
    def __init__(self, text): self.text = text
    def extract_text(self): return self.text


class TestDocumentTextExtraction(unittest.TestCase):
    def test_pdf_text_is_extracted_with_page_source_and_identifiers_redacted(self):
        fake_reader = type("Reader", (), {"is_encrypted": False, "pages": [FakePage("Name: Demo Person\\nPAN: ABCDE1234F\\nAccount Number: 123456789012") ]})()
        with patch("pypdf.PdfReader", return_value=fake_reader):
            result = extract_document_text(b"%PDF-fake", "application/pdf", "bank_statement")
        self.assertEqual(result["status"], "success")
        self.assertIn("[Page 1]", result["text"])
        self.assertNotIn("ABCDE1234F", result["text"])
        self.assertIn("[REDACTED PAN]", result["text"])
        self.assertIn("[REDACTED 12-DIGIT ID]", result["text"])

    def test_scanned_pdf_reports_ocr_unavailable_without_fabricating_text(self):
        fake_reader = type("Reader", (), {"is_encrypted": False, "pages": [FakePage("")]})()
        with patch("pypdf.PdfReader", return_value=fake_reader):
            result = extract_document_text(b"%PDF-fake", "application/pdf", "identity_pan")
        self.assertEqual(result["status"], "ocr_unavailable")
        self.assertEqual(result["text"], "")

    def test_image_extraction_reports_missing_tesseract(self):
        with patch("app.document_extraction.shutil.which", return_value=None):
            result = extract_document_text(b"fake-image", "image/png", "identity_pan")
        self.assertEqual(result["status"], "ocr_unavailable")
        self.assertIn("Tesseract", result["message"])

    def test_sensitive_identifier_masking_and_total_text_cap(self):
        masked = redact_sensitive_identifiers("IFSC: ABCD0123456 phone 9876543210")
        self.assertNotIn("ABCD0123456", masked)
        self.assertNotIn("9876543210", masked)
        docs = cap_document_texts([{"text": "a" * 8000}, {"text": "b" * 8000}])
        self.assertEqual(len(docs[0]["text"]) + len(docs[1]["text"]), 14000)
        self.assertTrue(docs[1]["truncated"])


if __name__ == "__main__":
    unittest.main()
