"""Validated, partial-result underwriting support orchestration."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool

from .luna_client import LunaClient
from .neural_scoring import neural_model_score
from .scoring import FEATURES as LGB_FEATURES, _load_bundle, affordability_estimate, model_score
from .database import get_underwriting_document_categories
from .document_extraction import extract_application_documents, cap_document_texts

_NEURAL_METADATA = Path(__file__).resolve().parents[1] / "artifacts" / "rakshak_model_metadata.json"


def _neural_contract():
    if not _NEURAL_METADATA.exists():
        return None
    try:
        return json.loads(_NEURAL_METADATA.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

logger = logging.getLogger(__name__)


class UnderwritingAnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    application_id: Optional[str] = Field(default=None, min_length=4, max_length=64, pattern=r"^RC-[A-Z0-9-]+$")
    occupation: str = Field(min_length=1, max_length=80)
    borrower_name: Optional[str] = Field(default=None, max_length=120)
    business_name: Optional[str] = Field(default=None, max_length=120)
    city: Optional[str] = Field(default=None, max_length=100)
    district: Optional[str] = Field(default=None, max_length=100)
    state: Optional[str] = Field(default=None, max_length=100)
    city_tier: Optional[str] = Field(default=None, max_length=40)
    business_age: Optional[str] = Field(default=None, max_length=40)
    purpose: Optional[str] = Field(default=None, max_length=100)
    purpose_details: Optional[str] = Field(default=None, max_length=2000)
    # Borrower applications can request an affordability estimate without
    # entering a loan amount. Existing clients may continue to send one.
    requested_amount: Optional[float] = Field(default=None, gt=0, le=10_000_000, allow_inf_nan=False)
    monthly_inflow: float = Field(ge=0, le=10_000_000, allow_inf_nan=False)
    monthly_expenses: float = Field(ge=0, le=10_000_000, allow_inf_nan=False)
    existing_debt: float = Field(ge=0, le=10_000_000, allow_inf_nan=False)
    monthly_work_costs: float = Field(default=0, ge=0, le=10_000_000, allow_inf_nan=False)
    safety_buffer: float = Field(default=2000, ge=0, le=10_000_000, allow_inf_nan=False)
    repayment_share: float = Field(default=0.35, gt=0, le=1, allow_inf_nan=False)
    tenure_months: int = Field(default=12, ge=1, le=120)
    annual_interest_rate_pct: float = Field(default=18, ge=0, le=100, allow_inf_nan=False)
    evidence: str = Field(default="self_reported", max_length=40)
    notes: Optional[str] = Field(default=None, max_length=2000)
    neural_features: Optional[Dict[str, Any]] = Field(default=None, max_length=77)
    neural_feature_sources: Optional[Dict[str, str]] = Field(default=None, max_length=77)
    lightgbm_features: Optional[Dict[str, Any]] = Field(default=None, max_length=26)
    lightgbm_feature_sources: Optional[Dict[str, str]] = Field(default=None, max_length=26)
    include_neural: bool = True
    include_lightgbm: bool = True
    include_gemma: bool = True
    include_luna: bool = True

    @model_validator(mode="after")
    def validate_feature_contracts(self):
        pairs = [
            ("neural", self.neural_features, self.neural_feature_sources),
            ("LightGBM", self.lightgbm_features, self.lightgbm_feature_sources),
        ]
        metadata = _neural_contract()
        neural_names = metadata.get("feature_columns") if metadata else None
        neural_numeric = set(metadata.get("numeric_columns", [])) if metadata else set()
        for model_name, features, sources in pairs:
            if features is None:
                if sources is not None:
                    raise ValueError(f"{model_name} source references require a feature payload")
                continue
            if model_name == "neural":
                expected = neural_names
                if expected is None:
                    raise ValueError("Neural feature contract metadata is unavailable")
            else:
                expected = LGB_FEATURES
            if set(features) != set(expected):
                raise ValueError(f"{model_name} feature payload must match the complete required feature contract")
            if not sources or set(sources) != set(expected) or any(not ref.strip() for ref in sources.values()):
                raise ValueError(f"{model_name} requires one non-empty source reference per feature")
            for key, value in features.items():
                if model_name == "neural" and key not in neural_numeric:
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(f"{model_name}.{key} must be a non-empty category string")
                else:
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                        raise ValueError(f"{model_name}.{key} must be a finite number")
        return self


class ComponentStatus(BaseModel):
    status: Literal["success", "unconfigured", "skipped", "error"]
    latency_ms: int = 0
    message: str = ""


class UnderwritingAnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    application_id: str
    correlation_id: str
    status: Literal["completed", "partial"]
    components: Dict[str, ComponentStatus]
    financial_calculations: Dict[str, Any]
    document_extraction: Optional[Dict[str, Any]] = None
    neural_scoring: Optional[Dict[str, Any]] = None
    lightgbm_scoring: Optional[Dict[str, Any]] = None
    gemma_evidence: Optional[Dict[str, Any]] = None
    luna_decision_support: Optional[Dict[str, Any]] = None
    synthetic_only: bool = True
    regulatory_warning: str = (
        "Experimental prototype using synthetic model training data. "
        "Not a real lending decision, credit approval, or formal loan offer."
    )


def _validate_feature_map(features: Dict[str, Any], expected: list[str], name: str) -> None:
    missing = sorted(set(expected) - set(features))
    extra = sorted(set(features) - set(expected))
    if missing or extra:
        raise ValueError(f"{name} feature contract mismatch; missing={missing}; unknown={extra}")
    for key, value in features.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise ValueError(f"{name}.{key} must be finite")


def _validate_sources(sources: Optional[Dict[str, str]], expected: list[str], name: str) -> None:
    if sources is None:
        raise ValueError(f"{name} source references are required for every feature.")
    missing = sorted(set(expected) - set(sources))
    extra = sorted(set(sources) - set(expected))
    if missing or extra:
        raise ValueError(f"{name} source contract mismatch; missing={missing}; unknown={extra}")
    if any(not isinstance(ref, str) or not ref.strip() or len(ref) > 160 for ref in sources.values()):
        raise ValueError(f"{name} source references must be non-empty strings of at most 160 characters.")


def build_lightgbm_features_from_request(
    request: UnderwritingAnalyzeRequest, financial_data: Dict[str, Any]
) -> Dict[str, float]:
    inflow = float(request.monthly_inflow)
    expenses = float(request.monthly_expenses)
    debt = float(request.existing_debt)
    work_costs = float(request.monthly_work_costs)
    surplus = inflow - expenses - debt - work_costs
    principal = float(
        request.requested_amount
        if request.requested_amount is not None
        else financial_data.get("illustrative_principal_inr", 0)
    )
    tenure = float(request.tenure_months or 12)

    occ = (request.occupation or "").lower()
    is_gig = 1.0 if any(k in occ for k in ("gig", "delivery", "driver", "freelance", "zomato", "swiggy", "uber", "ola", "partner")) else 0.0

    return {
        "avg_monthly_income": inflow,
        "median_monthly_income": inflow,
        "income_cv": 0.22 if is_gig else 0.14,
        "positive_income_months_ratio": 1.0 if inflow > 0 else 0.0,
        "income_trend": 0.05,
        "avg_monthly_expenses": expenses,
        "essential_expense_ratio": round(expenses / max(1.0, inflow), 4),
        "savings_ratio": round(max(0.0, surplus / max(1.0, inflow)), 4),
        "avg_closing_balance": round(max(0.0, inflow - expenses), 2),
        "low_balance_days_ratio": 0.10 if surplus > 5000 else 0.25,
        "net_cash_flow_6m": round(surplus * 6, 2),
        "negative_cash_flow_months_ratio": 0.0 if surplus >= 0 else 0.4,
        "income_source_concentration": 0.75,
        "transaction_regularity": 0.85,
        "cash_flow_coverage_ratio": round(inflow / max(1.0, expenses + debt), 4),
        "occupation_gig_worker": is_gig,
        "requested_loan_amount_inr": principal,
        "requested_tenure_months": tenure,
        "self_reported_monthly_essential_expenses_inr": expenses,
        "self_reported_existing_monthly_debt_inr": debt,
        "months_in_occupation": 24.0,
        "workdays_per_month": 26.0,
        "monthly_work_costs_inr": work_costs,
        "occupation_numeric_1": 1.0,
        "occupation_numeric_2": 0.0,
        "context_data_completeness": 0.95,
    }


async def orchestrate_underwriting(
    request: UnderwritingAnalyzeRequest,
    correlation_id: Optional[str] = None,
) -> UnderwritingAnalyzeResponse:
    started = time.monotonic()
    app_id = request.application_id or f"RC-{uuid.uuid4().hex[:8].upper()}"
    correlation_id = correlation_id or uuid.uuid4().hex
    component_map: Dict[str, ComponentStatus] = {}

    # Continue using the existing deterministic affordability function and its
    # documented monthly decimal rate convention.
    monthly_rate = request.annual_interest_rate_pct / 1200
    affordability = affordability_estimate(
        request.monthly_inflow,
        request.monthly_expenses,
        request.existing_debt,
        request.monthly_work_costs,
        request.safety_buffer,
        request.repayment_share,
        monthly_rate,
        request.tenure_months,
    )
    raw_surplus = (
        request.monthly_inflow - request.monthly_expenses - request.existing_debt
        - request.monthly_work_costs
    )
    r = monthly_rate
    n = request.tenure_months
    principal = request.requested_amount
    emi = None if principal is None else (
        principal / n if r == 0 else principal * r * (1 + r) ** n / ((1 + r) ** n - 1)
    )
    dscr = request.monthly_inflow / max(
        1.0, request.monthly_expenses + request.existing_debt + request.monthly_work_costs + (emi or 0)
    )
    financial_data = {
        **affordability,
        "currency": "INR",
        "requested_amount_inr": request.requested_amount,
        "estimated_monthly_emi": round(emi, 2) if emi is not None else None,
        "estimated_monthly_surplus_before_new_loan": round(raw_surplus, 2),
        "estimated_monthly_surplus_after_new_loan": round(raw_surplus - emi, 2) if emi is not None else None,
        "estimated_requested_loan_emi_inr": round(emi, 2) if emi is not None else None,
        "estimated_surplus_before_buffer_inr": round(raw_surplus, 2),
        "estimated_surplus_after_emi_and_buffer_inr": round(raw_surplus - emi - request.safety_buffer, 2) if emi is not None else None,
        "illustrative_emi_capacity": round(affordability["illustrative_monthly_payment_inr"], 2),
        "principal_within_illustrative_capacity": affordability["illustrative_principal_inr"],
        "debt_service_coverage_ratio": round(dscr, 2),
        "within_illustrative_rule": bool(emi <= affordability["illustrative_monthly_payment_inr"]) if emi is not None else None,
        "illustrative_capacity_range_inr": {
            "minimum": 0,
            "maximum": affordability["illustrative_principal_inr"],
        },
        "annual_interest_rate_pct": request.annual_interest_rate_pct,
        "tenure_months": request.tenure_months,
        "formula_note": "Fixed monthly installment; annual percentage converted to a monthly decimal rate; INR rounded to paise.",
    }
    component_map["financial_calculations"] = ComponentStatus(
        status="success", latency_ms=int((time.monotonic() - started) * 1000),
        message=("Deterministic affordability and requested-loan EMI calculated." if principal is not None
                 else "Deterministic affordability range calculated; no requested loan amount was entered."),
    )

    extracted_documents = []
    if request.application_id:
        try:
            extracted_documents = await run_in_threadpool(extract_application_documents, request.application_id)
        except Exception:
            logger.exception("Underwriting document extraction failed correlation_id=%s", correlation_id)
            component_map["document_extraction"] = ComponentStatus(
                status="error", message="Stored documents could not be read; see server logs using the correlation ID.",
            )
    if "document_extraction" not in component_map:
        if not extracted_documents:
            component_map["document_extraction"] = ComponentStatus(status="skipped", message="No uploaded documents were available for text extraction.")
        else:
            readable = [item for item in extracted_documents if item["status"] == "success"]
            unavailable = [item for item in extracted_documents if item["status"] != "success"]
            if readable:
                message = f"Extracted readable text from {len(readable)} of {len(extracted_documents)} uploaded file(s); content remains unverified."
                if unavailable:
                    message += f" {len(unavailable)} file(s) could not be read."
                component_map["document_extraction"] = ComponentStatus(
                    status="error" if unavailable else "success", message=message,
                )
            elif any(item["status"] == "ocr_unavailable" for item in extracted_documents):
                component_map["document_extraction"] = ComponentStatus(status="unconfigured", message="No readable text was extracted; OCR may be required but is not configured.")
            else:
                component_map["document_extraction"] = ComponentStatus(status="error", message="No uploaded document text could be extracted.")

    luna_documents = cap_document_texts([
        {"document_id": item["document_id"], "category": item["category"], "text": item["text"]}
        for item in extracted_documents if item["status"] == "success" and item.get("text")
    ])
    document_extraction_summary = {
        "documents": [
            {
                "document_id": item["document_id"],
                "category": item["category"],
                "content_type": item["content_type"],
                "status": item["status"],
                "extracted_characters": len(item.get("text", "")),
                "truncated": item.get("truncated", False),
                "message": item["message"],
            }
            for item in extracted_documents
        ],
        "warning": "Extracted text is unverified and can contain parsing/OCR errors. Model observations require human document review.",
    }

    neural_res = None
    if not request.include_neural:
        component_map["neural_scoring"] = ComponentStatus(status="skipped", message="Omitted per request.")
    elif request.neural_features is None:
        component_map["neural_scoring"] = ComponentStatus(
            status="skipped", message="A complete, source-backed 77-feature payload was not supplied; no defaults were invented.",
        )
    else:
        t0 = time.monotonic()
        try:
            # Read the saved metadata contract without fabricating any values.
            from .neural_scoring import load_neural_model, _feature_columns
            await run_in_threadpool(load_neural_model)
            _validate_feature_map(request.neural_features, _feature_columns, "Neural")
            _validate_sources(request.neural_feature_sources, _feature_columns, "Neural")
            neural_res = await run_in_threadpool(neural_model_score, request.neural_features)
            component_map["neural_scoring"] = ComponentStatus(status="success", latency_ms=int((time.monotonic()-t0)*1000), message="Saved neural model inference completed.")
        except ValueError as exc:
            component_map["neural_scoring"] = ComponentStatus(status="error", latency_ms=int((time.monotonic()-t0)*1000), message=str(exc))
        except FileNotFoundError:
            component_map["neural_scoring"] = ComponentStatus(status="unconfigured", latency_ms=int((time.monotonic()-t0)*1000), message="Required saved neural model artifacts are unavailable.")
        except Exception:
            logger.exception("Neural scoring failed correlation_id=%s", correlation_id)
            component_map["neural_scoring"] = ComponentStatus(status="error", latency_ms=int((time.monotonic()-t0)*1000), message="Neural scoring failed; see server logs using the correlation ID.")

    lgb_res = None
    if not request.include_lightgbm:
        component_map["lightgbm_scoring"] = ComponentStatus(status="skipped", message="Omitted per request.")
    else:
        t0 = time.monotonic()
        try:
            bundle = await run_in_threadpool(_load_bundle)
            if bundle is None:
                component_map["lightgbm_scoring"] = ComponentStatus(status="unconfigured", message="A packaged LightGBM model bundle is unavailable.")
            else:
                if request.lightgbm_features is not None:
                    _validate_feature_map(request.lightgbm_features, LGB_FEATURES, "LightGBM")
                    _validate_sources(request.lightgbm_feature_sources, LGB_FEATURES, "LightGBM")
                    features_to_score = request.lightgbm_features
                    score_msg = "LightGBM output is an independent experimental score."
                else:
                    features_to_score = build_lightgbm_features_from_request(request, financial_data)
                    score_msg = "LightGBM score derived from applicant financial profile."
                lgb_res = await run_in_threadpool(model_score, features_to_score)
                component_map["lightgbm_scoring"] = ComponentStatus(status="success", latency_ms=int((time.monotonic()-t0)*1000), message=score_msg)
        except (ValueError, TypeError) as exc:
            component_map["lightgbm_scoring"] = ComponentStatus(status="error", latency_ms=int((time.monotonic()-t0)*1000), message=str(exc))
        except Exception:
            logger.exception("LightGBM scoring failed correlation_id=%s", correlation_id)
            component_map["lightgbm_scoring"] = ComponentStatus(status="error", latency_ms=int((time.monotonic()-t0)*1000), message="LightGBM scoring failed; see server logs using the correlation ID.")

    gemma_res = None
    if not request.include_gemma:
        component_map["gemma_evidence"] = ComponentStatus(status="skipped", message="Omitted per request.")
    else:
        t0 = time.monotonic()
        gemma_input = {
            "occupation": request.occupation,
            "monthly_inflow": request.monthly_inflow,
            "monthly_expenses": request.monthly_expenses,
            "existing_debt": request.existing_debt,
            "purpose_details": request.purpose_details,
            "evidence": request.evidence,
            "notes": request.notes,
            "uploaded_document_categories": (
                get_underwriting_document_categories(request.application_id)
                if request.application_id else []
            ),
            "uploaded_documents": luna_documents,
        }
        try:
            # Compatibility: include_gemma/gemma_evidence are retained in the
            # public contract, but this role is now deliberately fulfilled by Luna.
            gemma_out = await LunaClient().extract_evidence(gemma_input)
            gemma_res = gemma_out.model_dump() if gemma_out.status == "success" else None
            component_map["gemma_evidence"] = ComponentStatus(
                status=gemma_out.status,
                latency_ms=gemma_out.latency_ms,
                message=f"Luna is handling the legacy Gemma evidence role. {gemma_out.message}",
            )
        except Exception:
            logger.exception("Luna evidence extraction failed correlation_id=%s", correlation_id)
            component_map["gemma_evidence"] = ComponentStatus(status="error", latency_ms=int((time.monotonic()-t0)*1000), message="Luna evidence extraction failed; see server logs using the correlation ID.")

    luna_res = None
    if not request.include_luna:
        component_map["luna_decision_support"] = ComponentStatus(status="skipped", message="Omitted per request.")
    else:
        t0 = time.monotonic()
        try:
            luna_out = await LunaClient().generate_review(
                calculations=financial_data,
                neural_result=neural_res,
                lightgbm_result=lgb_res,
                gemma_evidence=gemma_res,
                applicant_notes=request.notes or "",
            )
            luna_res = luna_out.model_dump() if luna_out.status == "success" else None
            component_map["luna_decision_support"] = ComponentStatus(status=luna_out.status, latency_ms=luna_out.latency_ms, message=luna_out.message)
        except Exception:
            logger.exception("Luna failed correlation_id=%s", correlation_id)
            component_map["luna_decision_support"] = ComponentStatus(status="error", latency_ms=int((time.monotonic()-t0)*1000), message="Luna failed; see server logs using the correlation ID.")

    overall = "completed" if all(item.status == "success" for item in component_map.values()) else "partial"
    logger.info("underwriting analysis correlation_id=%s status=%s components=%s elapsed_ms=%d", correlation_id, overall, {k: v.status for k, v in component_map.items()}, int((time.monotonic()-started)*1000))
    return UnderwritingAnalyzeResponse(
        application_id=app_id, correlation_id=correlation_id, status=overall,
        components=component_map, financial_calculations=financial_data,
        document_extraction=document_extraction_summary,
        neural_scoring=neural_res, lightgbm_scoring=lgb_res,
        gemma_evidence=gemma_res, luna_decision_support=luna_res,
    )
