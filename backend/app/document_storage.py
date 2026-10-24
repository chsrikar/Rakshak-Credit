"""Encrypted local storage for borrower evidence uploads."""
from __future__ import annotations

import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


def _cipher() -> Fernet:
    key = os.environ.get("DOCUMENT_ENCRYPTION_KEY", "").strip()
    if not key:
        raise RuntimeError("Encrypted document storage is not configured.")
    try:
        return Fernet(key.encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise RuntimeError("Document encryption configuration is invalid.") from exc


def encrypt_document(content: bytes) -> bytes:
    return _cipher().encrypt(content)


def decrypt_document(content: bytes) -> bytes:
    try:
        return _cipher().decrypt(content)
    except InvalidToken as exc:
        raise RuntimeError("Document could not be decrypted with the configured key.") from exc
