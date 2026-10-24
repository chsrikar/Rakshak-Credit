"""Run Luna's extraction and explanation roles, then the unified API."""

import asyncio
import os
import sys

from dotenv import load_dotenv

from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
load_dotenv(BACKEND_DIR / ".env", override=False)


def require_configuration() -> bool:
    required = {
        "LUNA_MODEL_NAME": os.getenv("LUNA_MODEL_NAME"),
        "LUNA_API_KEY or OPENAI_API_KEY": os.getenv("LUNA_API_KEY") or os.getenv("OPENAI_API_KEY"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        print("Live smoke tests not run; missing configuration: " + ", ".join(missing), file=sys.stderr)
        return False
    return True


async def run() -> int:
    if not require_configuration():
        return 2

    from app.luna_client import LunaClient
    from fastapi.testclient import TestClient
    from app.main import app

    fixture = {
        "occupation": "smoke-test example business",
        "monthly_inflow": 50000,
        "monthly_expenses": 30000,
        "existing_debt": 5000,
        "purpose_details": "Test-only example",
        "evidence": "self_reported",
        "notes": "Synthetic integration test input; no documents are attached.",
    }
    luna_client = LunaClient()
    evidence = await luna_client.extract_evidence(fixture)
    print(f"Luna evidence-extraction role: {evidence.status} (latency_ms={evidence.latency_ms}): {evidence.message}")
    evidence_ok = evidence.status == "success" and evidence.extraction is not None

    luna = await LunaClient().generate_review(
        calculations={
            "estimated_monthly_emi": 1000.0,
            "estimated_requested_loan_emi_inr": 1000.0,
            "estimated_monthly_surplus_before_new_loan": 15000.0,
            "estimated_surplus_before_buffer_inr": 15000.0,
            "estimated_surplus_after_emi_and_buffer_inr": 13000.0,
        },
        gemma_evidence=evidence.model_dump() if evidence_ok else None,
        applicant_notes=fixture["notes"],
    )
    print(f"Luna live request: {luna.status} (latency_ms={luna.latency_ms}): {luna.message}")
    luna_ok = luna.status == "success" and luna.decision_support is not None

    unified = {
        "borrower_name": "Integration Smoke Test",
        "occupation": fixture["occupation"],
        "business_name": "Synthetic Example",
        "city": "Test City",
        "district": "Test District",
        "state": "Test State",
        "city_tier": "tier2",
        "business_age": "2-5",
        "purpose": "inventory",
        "purpose_details": fixture["purpose_details"],
        "monthly_inflow": 50000,
        "monthly_expenses": 30000,
        "existing_debt": 5000,
        "requested_amount": 10000,
        "tenure_months": 12,
        "annual_interest_rate_pct": 18,
        "notes": fixture["notes"],
        "evidence": "synthetic_smoke_test",
        "include_neural": False,
        "include_lightgbm": False,
        "include_gemma": True,
        "include_luna": True,
    }
    with TestClient(app) as client:
        response = client.post("/api/v1/underwriting/analyze", json=unified)
    print(f"Unified underwriting API: HTTP {response.status_code}")
    if response.status_code != 200:
        return 1
    result = response.json()
    evidence_component = result["components"]["gemma_evidence"]
    luna_component = result["components"]["luna_decision_support"]
    evidence_status = evidence_component["status"]
    luna_status = luna_component["status"]
    print(f"Unified provider statuses: Luna evidence={evidence_status}, Luna explanation={luna_status}")
    statuses = result["components"]
    print(
        "Unified analysis: " + result["status"]
        + f"; financial={statuses['financial_calculations']['status']}"
        + f"; neural={statuses['neural_scoring']['status']}"
        + f"; LightGBM={statuses['lightgbm_scoring']['status']}"
    )
    if evidence_status != "success" or luna_status != "success" or not evidence_ok or not luna_ok:
        return 1
    print("Live Luna and unified API smoke tests passed; model content and secrets were not printed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
