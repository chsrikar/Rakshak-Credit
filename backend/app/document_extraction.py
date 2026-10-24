"""Bounded text extraction from encrypted underwriting evidence files."""
from __future__ import annotations

import io
import re
import shutil
from typing import Any

MAX_PAGES = 40
MAX_CHARS_PER_DOCUMENT = 6000
MAX_TOTAL_CHARS = 14000


def redact_sensitive_identifiers(text: str) -> str:
    """Mask common identity/account identifiers before exposing text to a model."""
    text = re.sub(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", "[REDACTED PAN]", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d{12}\b", "[REDACTED 12-DIGIT ID]", text)
    text = re.sub(r"(?i)(account\s*(?:number|no\.?|#)\s*[:\-]?\s*)\d{6,18}\b", r"\1[REDACTED ACCOUNT]", text)
    text = re.sub(r"(?i)(ifsc\s*(?:code)?\s*[:\-]?\s*)[A-Z]{4}0[A-Z0-9]{6}\b", r"\1[REDACTED IFSC]", text)
    text = re.sub(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)", "[REDACTED PHONE]", text)
    return text


def extract_document_text(content: bytes, content_type: str, category: str) -> dict[str, Any]:
    """Extract a bounded text preview; never claims that source data is verified."""
    if content_type == "application/pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content), strict=False)
            if reader.is_encrypted:
                return {"status": "error", "text": "", "message": "The PDF is encrypted or password protected."}
            if len(reader.pages) > MAX_PAGES:
                return {"status": "error", "text": "", "message": f"PDF exceeds the {MAX_PAGES}-page extraction limit."}
            pieces = []
            for index, page in enumerate(reader.pages, start=1):
                page_text = (page.extract_text() or "").strip()
                if page_text:
                    pieces.append(f"[Page {index}]\n{page_text}")
            text = redact_sensitive_identifiers("\n\n".join(pieces))[:MAX_CHARS_PER_DOCUMENT]
            if not text.strip():
                return {"status": "ocr_unavailable", "text": "", "message": "No selectable text was found; this may be a scanned PDF and requires OCR."}
            return {"status": "success", "text": text, "message": "Selectable PDF text extracted. This text is unverified and may contain extraction errors."}
        except Exception as exc:
            return {"status": "error", "text": "", "message": f"PDF text extraction failed ({type(exc).__name__})."}

    if content_type in {"image/png", "image/jpeg"}:
        if not shutil.which("tesseract"):
            return {"status": "ocr_unavailable", "text": "", "message": "Image OCR is unavailable: the Tesseract OCR runtime is not installed."}
        try:
            import pytesseract
            from PIL import Image
            text = pytesseract.image_to_string(Image.open(io.BytesIO(content)))
            text = redact_sensitive_identifiers(text)[:MAX_CHARS_PER_DOCUMENT]
            if not text.strip():
                return {"status": "error", "text": "", "message": "OCR completed but no readable text was detected."}
            return {"status": "success", "text": text, "message": "Image OCR text extracted. This text is unverified and may contain recognition errors."}
        except ImportError:
            return {"status": "ocr_unavailable", "text": "", "message": "Image OCR is unavailable: install pytesseract and the Tesseract OCR runtime."}
        except Exception as exc:
            return {"status": "error", "text": "", "message": f"Image OCR failed ({type(exc).__name__})."}

    return {"status": "unsupported", "text": "", "message": f"Unsupported content type for extraction: {category}."}


def cap_document_texts(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    remaining = MAX_TOTAL_CHARS
    result = []
    for item in documents:
        copied = dict(item)
        text = copied.get("text", "")
        copied["text"] = text[:remaining]
        remaining -= len(copied["text"])
        if len(copied["text"]) < len(text):
            copied["truncated"] = True
        result.append(copied)
    return result


def extract_application_documents(application_id: str) -> list[dict[str, Any]]:
    """Decrypt and extract a bounded set of documents for an authorized caller."""
    from .database import list_encrypted_underwriting_documents
    from .document_storage import decrypt_document

    records = list_encrypted_underwriting_documents(application_id)
    extracted = []
    total_bytes = 0
    for record in records:
        entry = {
            "document_id": record["id"],
            "category": record["category"],
            "content_type": record["content_type"],
        }
        size = int(record["size_bytes"])
        if total_bytes + size > 24 * 1024 * 1024:
            entry.update(status="error", text="", message="Application extraction limit reached; remaining files were not read.")
            extracted.append(entry)
            continue
        total_bytes += size
        try:
            plaintext = decrypt_document(record["encrypted_content"])
        except RuntimeError:
            entry.update(status="error", text="", message="Encrypted document could not be opened with the configured storage key.")
        else:
            entry.update(extract_document_text(plaintext, record["content_type"], record["category"]))
        extracted.append(entry)
    return extracted
