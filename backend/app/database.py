import sqlite3
from pathlib import Path
from contextlib import contextmanager
import json
from datetime import datetime, timezone
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "rakshak_demo.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

@contextmanager
def connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with connection() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS applicants (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, occupation TEXT NOT NULL, city TEXT NOT NULL,
            monthly_income REAL NOT NULL, monthly_expenses REAL NOT NULL, monthly_debt REAL NOT NULL,
            closing_balance REAL NOT NULL, transaction_regularity REAL NOT NULL,
            data_completeness REAL NOT NULL, demo_only INTEGER NOT NULL DEFAULT 1)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS underwriting_applications (
            id TEXT PRIMARY KEY, owner_id TEXT, payload_encrypted BLOB NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
        application_columns = {row["name"] for row in conn.execute("PRAGMA table_info(underwriting_applications)").fetchall()}
        if "owner_id" not in application_columns:
            conn.execute("ALTER TABLE underwriting_applications ADD COLUMN owner_id TEXT")
        if "review_status" not in application_columns:
            conn.execute("ALTER TABLE underwriting_applications ADD COLUMN review_status TEXT NOT NULL DEFAULT 'pending'")
        if "reviewed_by" not in application_columns:
            conn.execute("ALTER TABLE underwriting_applications ADD COLUMN reviewed_by TEXT")
        if "reviewed_at" not in application_columns:
            conn.execute("ALTER TABLE underwriting_applications ADD COLUMN reviewed_at TEXT")
        conn.execute("""CREATE TABLE IF NOT EXISTS underwriting_documents (
            id TEXT PRIMARY KEY,
            application_id TEXT NOT NULL REFERENCES underwriting_applications(id) ON DELETE CASCADE,
            category TEXT NOT NULL, stored_filename TEXT NOT NULL, content_type TEXT NOT NULL,
            size_bytes INTEGER NOT NULL, encrypted_content BLOB NOT NULL, created_at TEXT NOT NULL)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS underwriting_analyses (
            id TEXT PRIMARY KEY,
            application_id TEXT NOT NULL REFERENCES underwriting_applications(id) ON DELETE CASCADE,
            response_encrypted BLOB NOT NULL, created_at TEXT NOT NULL)""")

def list_applicants():
    with connection() as conn:
        rows = conn.execute("SELECT * FROM applicants ORDER BY id").fetchall()
    return [dict(r) for r in rows]

def get_applicant(applicant_id):
    with connection() as conn:
        row = conn.execute("SELECT * FROM applicants WHERE id=?", (applicant_id,)).fetchone()
    return dict(row) if row else None

def insert_applicants(rows):
    with connection() as conn:
        conn.executemany("""INSERT OR REPLACE INTO applicants
        (id,name,occupation,city,monthly_income,monthly_expenses,monthly_debt,closing_balance,
         transaction_regularity,data_completeness,demo_only)
        VALUES (:id,:name,:occupation,:city,:monthly_income,:monthly_expenses,:monthly_debt,
         :closing_balance,:transaction_regularity,:data_completeness,1)""", rows)

def save_underwriting_application(application_id, payload, owner_id=None):
    from .document_storage import encrypt_document
    now = datetime.now(timezone.utc).isoformat()
    encrypted_payload = encrypt_document(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    with connection() as conn:
        conn.execute("""INSERT INTO underwriting_applications
            (id, owner_id, payload_encrypted, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET payload_encrypted=excluded.payload_encrypted,
                updated_at=excluded.updated_at""",
            (application_id, owner_id, encrypted_payload, now, now))

def get_underwriting_application(application_id):
    with connection() as conn:
        row = conn.execute("SELECT * FROM underwriting_applications WHERE id=?", (application_id,)).fetchone()
    return dict(row) if row else None

def get_underwriting_application_payload(application_id):
    from .document_storage import decrypt_document
    application = get_underwriting_application(application_id)
    if not application:
        return None
    return json.loads(decrypt_document(application["payload_encrypted"]).decode("utf-8"))

def list_underwriting_applications():
    from .document_storage import decrypt_document
    with connection() as conn:
        rows = conn.execute("SELECT * FROM underwriting_applications ORDER BY updated_at DESC").fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["application"] = json.loads(decrypt_document(item.pop("payload_encrypted")).decode("utf-8"))
        result.append(item)
    return result

def list_applications_for_owner(owner_id):
    return [item for item in list_underwriting_applications() if item.get("owner_id") == owner_id]

def set_underwriting_review_status(application_id, status, reviewer):
    now = datetime.now(timezone.utc).isoformat()
    reviewed_at = now if status == "reviewed" else None
    with connection() as conn:
        cursor = conn.execute(
            "UPDATE underwriting_applications SET review_status=?, reviewed_by=?, reviewed_at=?, updated_at=? WHERE id=?",
            (status, reviewer, reviewed_at, now, application_id),
        )
    return cursor.rowcount > 0

def get_encrypted_underwriting_document(application_id, document_id):
    with connection() as conn:
        row = conn.execute("""SELECT * FROM underwriting_documents
            WHERE application_id=? AND id=?""", (application_id, document_id)).fetchone()
    return dict(row) if row else None

def list_encrypted_underwriting_documents(application_id):
    """Internal-only document access for extraction after route authorization."""
    with connection() as conn:
        rows = conn.execute("""SELECT id, application_id, category, stored_filename,
            content_type, size_bytes, created_at, encrypted_content FROM underwriting_documents
            WHERE application_id=? ORDER BY created_at, id LIMIT 10""", (application_id,)).fetchall()
    return [dict(row) for row in rows]

def list_underwriting_analyses(application_id):
    from .document_storage import decrypt_document
    with connection() as conn:
        rows = conn.execute("""SELECT id, response_encrypted, created_at FROM underwriting_analyses
            WHERE application_id=? ORDER BY created_at DESC""", (application_id,)).fetchall()
    return [{"id": row["id"], "created_at": row["created_at"],
             "analysis": json.loads(decrypt_document(row["response_encrypted"]).decode("utf-8"))}
            for row in rows]

def insert_underwriting_document(document):
    with connection() as conn:
        conn.execute("""INSERT INTO underwriting_documents
            (id, application_id, category, stored_filename, content_type, size_bytes, encrypted_content, created_at)
            VALUES (:id, :application_id, :category, :stored_filename, :content_type, :size_bytes, :encrypted_content, :created_at)""", document)

def list_underwriting_documents(application_id):
    with connection() as conn:
        rows = conn.execute("""SELECT id, application_id, category, stored_filename,
            content_type, size_bytes, created_at FROM underwriting_documents
            WHERE application_id=? ORDER BY created_at, id""", (application_id,)).fetchall()
    return [dict(row) for row in rows]

def get_underwriting_document_categories(application_id):
    with connection() as conn:
        rows = conn.execute("SELECT DISTINCT category FROM underwriting_documents WHERE application_id=?", (application_id,)).fetchall()
    return [row["category"] for row in rows]

def delete_underwriting_document(application_id, document_id):
    with connection() as conn:
        cursor = conn.execute(
            "DELETE FROM underwriting_documents WHERE application_id=? AND id=?",
            (application_id, document_id),
        )
    return cursor.rowcount > 0

def save_underwriting_analysis(application_id, response):
    from .document_storage import encrypt_document
    now = datetime.now(timezone.utc).isoformat()
    encrypted_response = encrypt_document(json.dumps(response, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    analysis_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    with connection() as conn:
        conn.execute("""INSERT INTO underwriting_analyses
            (id, application_id, response_encrypted, created_at) VALUES (?, ?, ?, ?)""",
            (analysis_id, application_id, encrypted_response, now))
