"""
Rakshak Credit — FastAPI backend
All data is synthetic. Not for real lending decisions.
"""
from contextlib import asynccontextmanager
from io import StringIO
from pathlib import Path
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .database import (
    init_db, list_applicants, get_applicant, save_underwriting_application,
    get_underwriting_application, insert_underwriting_document,
    list_underwriting_documents, delete_underwriting_document, save_underwriting_analysis,
    get_underwriting_application_payload, list_underwriting_applications,
    list_applications_for_owner,
    get_encrypted_underwriting_document, list_underwriting_analyses,
    get_underwriting_document_categories,
    set_underwriting_review_status,
)
from .document_storage import encrypt_document, decrypt_document
from .document_extraction import extract_document_text
from .auth import (
    Principal, authenticate, configuration_ready, current_user,
    issue_access_token, require_application_access, require_roles,
)
from .demo_data import seed_demo_data
from .schemas import EstimateRequest, ScoreRequest, NeuralScoreRequest
from .scoring import model_score, affordability_estimate, _load_bundle
from .neural_scoring import neural_model_score
from .underwriting_orchestrator import (
    UnderwritingAnalyzeRequest,
    UnderwritingAnalyzeResponse,
    ComponentStatus,
    orchestrate_underwriting,
)
from .features import FEATURES

MAX_UNDERWRITING_DOCUMENT_BYTES = 12 * 1024 * 1024
DOCUMENT_CATEGORIES = {
    "identity_pan", "bank_statement", "tax_return", "gst_record",
    "business_registration", "income_invoice", "platform_statement", "other",
}
REQUIRED_BORROWER_DOCUMENT_CATEGORIES = {
    "identity_pan", "bank_statement", "tax_return",
}


class DemoLoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=1, max_length=256)


class HumanReviewUpdate(BaseModel):
    status: Literal["in_review", "reviewed"]

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "data"
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ROOT_ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"


@asynccontextmanager
async def lifespan(app):
    init_db()
    seed_demo_data()
    yield


app = FastAPI(
    title="Rakshak Credit Demo API",
    version="2.0.0",
    description=(
        "Cash-flow credit scoring for credit-invisible workers. "
        "Synthetic-data prototype — not for real lending decisions."
    ),
    lifespan=lifespan,
)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$",
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Core endpoints ───────────────────────────────────────────────────


@app.post("/api/v1/auth/login")
def demo_login(payload: DemoLoginRequest, response: Response):
    if not configuration_ready():
        raise HTTPException(status_code=503, detail="Demo sign-in is not configured on this backend.")
    principal = authenticate(payload.username, payload.password)
    if not principal:
        raise HTTPException(status_code=401, detail="Username or password is incorrect.")
    expires = int(os.environ.get("AUTH_TOKEN_TTL_SECONDS", "28800"))
    response.set_cookie(
        "rakshak_session", issue_access_token(principal), max_age=expires,
        httponly=True, secure=os.environ.get("SESSION_COOKIE_SECURE", "false").lower() == "true",
        samesite="lax", path="/",
    )
    response.headers["Cache-Control"] = "no-store"
    return {
        "expires_in": expires,
        "user": {"username": principal.username, "role": principal.role},
    }


@app.get("/api/v1/auth/me")
def demo_current_user(principal: Principal = Depends(current_user)):
    return {"username": principal.username, "role": principal.role}


@app.post("/api/v1/auth/logout", status_code=204)
def demo_logout(response: Response):
    response.delete_cookie("rakshak_session", path="/", httponly=True, samesite="lax")
    response.headers["Cache-Control"] = "no-store"
    # Return no content; returning the injected Response as a response body
    # produces a malformed status line on some Starlette/Uvicorn versions.
    return None


@app.get("/")
def root():
    return {
        "name": "Rakshak Credit",
        "status": "ok",
        "synthetic_only": True,
        "message": (
            "Hackathon prototype; not a real credit decision system."
        ),
    }


@app.get("/health")
@app.get("/api/v1/health")
def health():
    model_exists = (ARTIFACT_DIR / "best_model.joblib").exists()
    return {
        "status": "healthy",
        "synthetic_only": True,
        "model_artifact_present": model_exists,
    }


# ── Applicant endpoints ─────────────────────────────────────────────


@app.get("/api/v1/applicants")
def applicants():
    rows = list_applicants()
    return {"count": len(rows), "items": rows, "synthetic_only": True}


@app.get("/api/v1/applicants/{applicant_id}")
def applicant_detail(applicant_id: str):
    row = get_applicant(applicant_id)
    if not row:
        raise HTTPException(status_code=404, detail="Applicant not found")
    return {**row, "synthetic_only": True}


@app.get("/api/v1/applicants/{applicant_id}/score")
def applicant_score(applicant_id: str, principal: Principal = Depends(require_roles("lender"))):
    """
    Score an applicant by loading their features from the dataset
    and running through the full scoring + SHAP pipeline.
    """
    features_path = DATA_DIR / "model_ready_dataset.csv"
    if not features_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Feature dataset not found. Run data generation first.",
        )

    df = pd.read_csv(features_path)
    applicant_row = df[df["applicant_id"] == applicant_id]

    if applicant_row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No features found for {applicant_id}",
        )

    row = applicant_row.iloc[0]
    feature_dict = {f: float(row[f]) for f in FEATURES if f in row.index}

    missing = [f for f in FEATURES if f not in feature_dict]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing features: {missing}",
        )

    result = model_score(feature_dict)
    result["applicant_id"] = applicant_id
    return result


# ── Scoring and estimation ───────────────────────────────────────────


@app.post("/api/v1/eligibility/estimate")
def estimate(payload: EstimateRequest, principal: Principal = Depends(current_user)):
    return affordability_estimate(
        payload.monthly_income,
        payload.essential_expenses,
        payload.existing_monthly_debt,
        payload.monthly_work_costs,
        payload.safety_buffer,
        payload.repayment_share,
        payload.monthly_interest_rate,
        payload.tenure_months,
    )


@app.post("/api/v1/score")
def score(payload: ScoreRequest, principal: Principal = Depends(current_user)):
    try:
        return model_score(payload.features)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/api/v1/neural-score")
def neural_score(payload: ScoreRequest, principal: Principal = Depends(current_user)):
    try:
        return neural_model_score(payload.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Neural scoring failed. Check backend logs."
        )


@app.post("/api/v1/underwriting/analyze", response_model=UnderwritingAnalyzeResponse)
async def analyze_underwriting(
    payload: UnderwritingAnalyzeRequest,
    request: Request,
    response: Response,
    principal: Principal = Depends(current_user),
):
    """
    Unified multi-model underwriting analysis:
    - Deterministic cash-flow arithmetic
    - Deep Tabular Neural Network (77 features)
    - LightGBM baseline scoring (26 features)
    - Luna evidence extraction (legacy Gemma response field)
    - Luna structured decision support synthesis
    """
    requested_id = request.headers.get("X-Correlation-ID", "")
    correlation_id = requested_id if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", requested_id) else uuid.uuid4().hex
    response.headers["X-Correlation-ID"] = correlation_id
    try:
        if payload.application_id:
            require_application_access(payload.application_id, principal)
            if principal.role == "borrower":
                uploaded_categories = set(get_underwriting_document_categories(payload.application_id))
                missing_categories = sorted(REQUIRED_BORROWER_DOCUMENT_CATEGORIES - uploaded_categories)
                if missing_categories:
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "message": "Upload all required borrower documents before requesting an estimate.",
                            "missing_document_categories": missing_categories,
                        },
                    )
            save_underwriting_application(
                payload.application_id,
                payload.model_dump(exclude={"application_id", "neural_features", "neural_feature_sources", "lightgbm_features", "lightgbm_feature_sources"}),
                owner_id=principal.username,
            )
        analysis = await orchestrate_underwriting(payload, correlation_id=correlation_id)
        if payload.application_id:
            try:
                save_underwriting_analysis(payload.application_id, analysis.model_dump(mode="json"))
                analysis.components["database_persistence"] = ComponentStatus(
                    status="success", message="Encrypted analysis result saved to the application record."
                )
            except Exception:
                logger.exception("Underwriting persistence failed correlation_id=%s", correlation_id)
                analysis.components["database_persistence"] = ComponentStatus(
                    status="error", message="Analysis completed but could not be saved to the application record."
                )
                analysis.status = "partial"
        return analysis
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled underwriting error correlation_id=%s", correlation_id)
        raise HTTPException(
            status_code=500,
            detail=f"Underwriting analysis failed. Reference {correlation_id}."
        )


@app.post("/api/v1/underwriting/applications", status_code=201)
def create_underwriting_application(
    payload: UnderwritingAnalyzeRequest,
    principal: Principal = Depends(require_roles("borrower")),
):
    application_id = f"RC-{uuid.uuid4().hex[:12].upper()}"
    safe_payload = payload.model_dump(exclude={"application_id", "neural_features", "neural_feature_sources", "lightgbm_features", "lightgbm_feature_sources"})
    save_underwriting_application(application_id, safe_payload, owner_id=principal.username)
    return {
        "application_id": application_id,
        "status": "draft_saved",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/underwriting/applications")
def borrower_application_list(principal: Principal = Depends(require_roles("borrower"))):
    items = []
    for item in list_applications_for_owner(principal.username):
        application_id = item["id"]
        items.append({
            "application_id": application_id,
            "created_at": item["created_at"],
            "updated_at": item["updated_at"],
            "application": item["application"],
            "documents": list_underwriting_documents(application_id),
            "analyses": list_underwriting_analyses(application_id),
        })
    return {"count": len(items), "items": items}


@app.get("/api/v1/underwriting/applications/{application_id}")
def underwriting_application_detail(
    application_id: str,
    principal: Principal = Depends(current_user),
):
    require_application_access(application_id, principal)
    application = get_underwriting_application(application_id)
    return {
        "application_id": application_id,
        "created_at": application["created_at"],
        "updated_at": application["updated_at"],
        "application": get_underwriting_application_payload(application_id),
        "documents": list_underwriting_documents(application_id),
        "analyses": list_underwriting_analyses(application_id),
    }


@app.put("/api/v1/underwriting/applications/{application_id}")
def update_underwriting_application(
    application_id: str,
    payload: UnderwritingAnalyzeRequest,
    principal: Principal = Depends(require_roles("borrower")),
):
    require_application_access(application_id, principal)
    if not get_underwriting_application(application_id):
        raise HTTPException(status_code=404, detail="Application was not found.")
    safe_payload = payload.model_dump(exclude={"application_id", "neural_features", "neural_feature_sources", "lightgbm_features", "lightgbm_feature_sources"})
    save_underwriting_application(application_id, safe_payload)
    return {
        "application_id": application_id,
        "status": "draft_saved",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/underwriting/applications/{application_id}/documents")
def underwriting_documents(
    application_id: str,
    principal: Principal = Depends(current_user),
):
    require_application_access(application_id, principal)
    if not get_underwriting_application(application_id):
        raise HTTPException(status_code=404, detail="Application was not found.")
    return {"application_id": application_id, "items": list_underwriting_documents(application_id)}


@app.delete("/api/v1/underwriting/applications/{application_id}/documents/{document_id}")
def remove_underwriting_document(
    application_id: str,
    document_id: str,
    principal: Principal = Depends(require_roles("borrower")),
):
    require_application_access(application_id, principal)
    if not get_underwriting_application(application_id):
        raise HTTPException(status_code=404, detail="Application was not found.")
    if not delete_underwriting_document(application_id, document_id):
        raise HTTPException(status_code=404, detail="Document was not found for this application.")
    return {"application_id": application_id, "document_id": document_id, "status": "removed"}


@app.post("/api/v1/underwriting/applications/{application_id}/documents", status_code=201)
async def upload_underwriting_document(
    application_id: str,
    file: UploadFile = File(...),
    category: str = Form(...),
    principal: Principal = Depends(require_roles("borrower")),
):
    require_application_access(application_id, principal)
    if not get_underwriting_application(application_id):
        raise HTTPException(status_code=404, detail="Application was not found.")
    if category not in DOCUMENT_CATEGORIES:
        raise HTTPException(status_code=422, detail="Choose a supported document category.")
    content = await file.read(MAX_UNDERWRITING_DOCUMENT_BYTES + 1)
    if not content:
        raise HTTPException(status_code=422, detail="The selected file is empty.")
    if len(content) > MAX_UNDERWRITING_DOCUMENT_BYTES:
        raise HTTPException(status_code=413, detail="Each document must be 12 MB or smaller.")

    if content.startswith(b"%PDF-"):
        content_type, extension = "application/pdf", ".pdf"
    elif content.startswith(b"\x89PNG\r\n\x1a\n"):
        content_type, extension = "image/png", ".png"
    elif content.startswith(b"\xff\xd8\xff"):
        content_type, extension = "image/jpeg", ".jpg"
    else:
        raise HTTPException(status_code=415, detail="Upload a valid PDF, PNG, or JPEG file.")

    try:
        encrypted = encrypt_document(content)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    filename = f"evidence-{uuid.uuid4().hex[:12]}{extension}"
    record = {
        "id": uuid.uuid4().hex,
        "application_id": application_id,
        "category": category,
        "stored_filename": filename,
        "content_type": content_type,
        "size_bytes": len(content),
        "encrypted_content": encrypted,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    insert_underwriting_document(record)
    record.pop("encrypted_content")
    return record


@app.get("/api/v1/underwriting/applications/{application_id}/documents/{document_id}/download")
def download_underwriting_document(
    application_id: str,
    document_id: str,
    principal: Principal = Depends(current_user),
):
    require_application_access(application_id, principal)
    record = get_encrypted_underwriting_document(application_id, document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document was not found for this application.")
    try:
        content = decrypt_document(record["encrypted_content"])
    except RuntimeError as exc:
        logger.exception("Document decryption failed application_id=%s document_id=%s", application_id, document_id)
        raise HTTPException(status_code=503, detail="Document storage key is unavailable or invalid.") from exc
    return Response(content=content, media_type=record["content_type"], headers={
        "Content-Disposition": f'attachment; filename="{record["stored_filename"]}"',
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    })


@app.get("/api/v1/underwriting/applications/{application_id}/documents/{document_id}/text")
def read_underwriting_document_text(
    application_id: str,
    document_id: str,
    principal: Principal = Depends(current_user),
):
    """Return bounded, PII-redacted extracted text to the application owner or lender."""
    require_application_access(application_id, principal)
    record = get_encrypted_underwriting_document(application_id, document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document was not found for this application.")
    try:
        content = decrypt_document(record["encrypted_content"])
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Document storage key is unavailable or invalid.") from exc
    extracted = extract_document_text(content, record["content_type"], record["category"])
    return {
        "application_id": application_id,
        "document_id": document_id,
        "category": record["category"],
        **extracted,
        "warning": "Text extraction is not verification. Review the original file; identifiers may be masked and extraction can be incomplete.",
    }


@app.get("/api/v1/lender/applications")
def lender_application_list(principal: Principal = Depends(require_roles("lender"))):
    rows = []
    for item in list_underwriting_applications():
        application_id = item["id"]
        rows.append({
            "application_id": application_id,
            "owner": item["owner_id"],
            "created_at": item["created_at"],
            "updated_at": item["updated_at"],
            "review_status": item.get("review_status", "pending"),
            "reviewed_by": item.get("reviewed_by"),
            "reviewed_at": item.get("reviewed_at"),
            "application": item["application"],
            "documents": list_underwriting_documents(application_id),
            "analyses": list_underwriting_analyses(application_id),
        })
    return {"count": len(rows), "items": rows}


@app.put("/api/v1/lender/applications/{application_id}/review")
def update_lender_review_status(
    application_id: str,
    payload: HumanReviewUpdate,
    principal: Principal = Depends(require_roles("lender")),
):
    if not set_underwriting_review_status(application_id, payload.status, principal.username):
        raise HTTPException(status_code=404, detail="Application was not found.")
    return {"application_id": application_id, "review_status": payload.status}


# ── CSV upload scoring ───────────────────────────────────────────────


@app.post("/api/v1/data/upload")
async def upload_csv(file: UploadFile = File(...), principal: Principal = Depends(require_roles("lender"))):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file.")

    try:
        frame = pd.read_csv(
            StringIO((await file.read()).decode("utf-8-sig"))
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not parse CSV: {exc}"
        )

    missing = [f for f in FEATURES if f not in frame.columns]
    if missing:
        raise HTTPException(
            status_code=422, detail={"missing_features": missing}
        )

    if len(frame) > 500:
        raise HTTPException(
            status_code=413, detail="Demo upload limit is 500 rows."
        )

    if frame[FEATURES].isna().any().any():
        raise HTTPException(
            status_code=422, detail="CSV contains missing feature values."
        )

    scores = []
    for idx, row in frame[FEATURES].iterrows():
        try:
            result = model_score({f: float(row[f]) for f in FEATURES})
            scores.append({"row": int(idx) + 1, **result})
        except Exception as exc:
            raise HTTPException(
                status_code=422, detail=f"Row {idx + 1}: {exc}"
            )

    return {
        "rows_received": len(frame),
        "scores": scores,
        "synthetic_only": True,
    }


# ── Model metrics and explainability ─────────────────────────────────


@app.get("/api/v1/model/metrics")
def model_metrics():
    """Return training metrics from the artifacts."""
    metrics_path = ROOT_ARTIFACT_DIR / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(
            status_code=503, detail="Metrics not found. Train the model first."
        )
    with open(metrics_path) as f:
        return json.load(f)


@app.get("/api/v1/model/feature-importance")
def feature_importance():
    """Return SHAP-based global feature importance."""
    shap_path = ROOT_ARTIFACT_DIR / "shap_global_importance.csv"
    if not shap_path.exists():
        raise HTTPException(
            status_code=503,
            detail="SHAP importance file not found.",
        )
    df = pd.read_csv(shap_path)
    return {
        "features": df.to_dict(orient="records"),
        "synthetic_only": True,
    }


# ── Synthetic model-score comparison demo ────────────────────────────


@app.get("/api/v1/demo/comparison")
def comparison_demo():
    """
    Compare two synthetic records by experimental score only. This route
    does not issue bureau, approval, or rejection decisions.
    """
    features_path = DATA_DIR / "model_ready_dataset.csv"
    if not features_path.exists():
        raise HTTPException(
            status_code=503, detail="Feature dataset not found."
        )

    df = pd.read_csv(features_path)

    # Find two applicants: one low-risk, one higher-risk, similar income
    df_sorted = df.sort_values("avg_monthly_income")

    # Pick from the middle income range for realism
    mid = len(df_sorted) // 2
    candidates = df_sorted.iloc[max(0, mid - 50): mid + 50].copy()

    # Fast scoring for selection
    bundle = _load_bundle()
    if bundle is None:
        raise HTTPException(status_code=503, detail="Packaged scoring model is unavailable.")
    model = bundle["model"]
    X_mat = candidates[FEATURES]
    probs = model.predict_proba(X_mat)[:, 1]

    candidates = candidates.copy()
    candidates["risk_prob"] = probs
    candidates = candidates.sort_values("risk_prob")

    lower_score_row = candidates.iloc[0]
    higher_score_row = candidates.iloc[-1]

    lower_score_feat = {f: float(lower_score_row[f]) for f in FEATURES if f in lower_score_row.index}
    higher_score_feat = {f: float(higher_score_row[f]) for f in FEATURES if f in higher_score_row.index}

    lower_score_result = model_score(lower_score_feat)
    higher_score_result = model_score(higher_score_feat)

    lower_score_id = lower_score_row["applicant_id"]
    higher_score_id = higher_score_row["applicant_id"]

    lower_score_info = get_applicant(lower_score_id) or {
        "id": lower_score_id, "name": "Synthetic demo applicant", "occupation": "Shopkeeper", "city": lower_score_row.get("city", "Unknown")
    }
    higher_score_info = get_applicant(higher_score_id) or {
        "id": higher_score_id, "name": "Synthetic demo applicant", "occupation": "Shopkeeper", "city": higher_score_row.get("city", "Unknown")
    }

    return {
        "title": "Synthetic model-score comparison",
        "narrative": (
            "This compares two rows from the synthetic demo dataset using the packaged model. "
            "Scores are experimental, do not represent verified transaction evidence or bureau outcomes, "
            "and must not be interpreted as approvals or rejections."
        ),
        "lower_score_applicant": {
            "info": lower_score_info,
            "score": lower_score_result,
            "features_summary": {
                "monthly_income": round(float(lower_score_feat.get("avg_monthly_income", 0)), 2),
                "income_volatility": round(float(lower_score_feat.get("income_cv", 0)), 3),
                "low_balance_days": f"{round(float(lower_score_feat.get('low_balance_days_ratio', 0)) * 100)}%",
                "transaction_regularity": f"{round(float(lower_score_feat.get('transaction_regularity', 0)) * 100)}%",
                "avg_closing_balance": round(float(lower_score_feat.get("avg_closing_balance", 0)), 2),
            }
        },
        "higher_score_applicant": {
            "info": higher_score_info,
            "score": higher_score_result,
            "features_summary": {
                "monthly_income": round(float(higher_score_feat.get("avg_monthly_income", 0)), 2),
                "income_volatility": round(float(higher_score_feat.get("income_cv", 0)), 3),
                "low_balance_days": f"{round(float(higher_score_feat.get('low_balance_days_ratio', 0)) * 100)}%",
                "transaction_regularity": f"{round(float(higher_score_feat.get('transaction_regularity', 0)) * 100)}%",
                "avg_closing_balance": round(float(higher_score_feat.get("avg_closing_balance", 0)), 2),
            }
        },
        "synthetic_only": True,
    }


# ── Fairness scaffold ───────────────────────────────────────────────


@app.get("/api/v1/fairness/metrics")
def fairness_metrics():
    return {
        "status": "not_evaluated",
        "synthetic_only": True,
        "message": (
            "No valid fairness conclusion can be drawn from "
            "the included synthetic demo data."
        ),
        "approach": {
            "excluded_from_model": [
                "gender", "religion", "caste", "region",
            ],
            "audit_dimensions": [
                "occupation_type", "city", "income_quartile",
            ],
            "checks": [
                "Approval-rate parity across groups",
                "False-rejection rate gaps",
                "Score calibration by group",
            ],
        },
        "recommended_checks": [
            "Compare error rates across appropriate groups using lawful data.",
            "Check calibration and selection-rate differences with uncertainty intervals.",
            "Document limitations and seek independent review.",
        ],
    }
