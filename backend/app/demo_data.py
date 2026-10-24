"""
Seed demo applicants from the real synthetic dataset
instead of random generation. Loads from data/data/ CSVs,
picks representative profiles, and computes features for each.
"""
import random
from pathlib import Path
import pandas as pd
import numpy as np
from .database import list_applicants, insert_applicants

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "data"


def _load_real_profiles():
    """
    Load actual synthetic applicant profiles and their model-ready features.
    Returns a list of applicant dicts ready for the database.
    """
    applicants_path = DATA_DIR / "applicants.csv"
    features_path = DATA_DIR / "model_ready_dataset.csv"

    if not applicants_path.exists() or not features_path.exists():
        return _fallback_profiles()

    apps = pd.read_csv(applicants_path)
    feats = pd.read_csv(features_path)

    # Merge on applicant_id
    merged = apps.merge(feats, on="applicant_id", how="inner")

    # Pick a representative sample: 15 gig workers + 15 shopkeepers
    gig = merged[merged["occupation_type"] == "gig_worker"]
    shop = merged[merged["occupation_type"] == "small_shopkeeper"]

    rng = random.Random(42)
    gig_sample = gig.sample(n=min(15, len(gig)), random_state=42)
    shop_sample = shop.sample(n=min(15, len(shop)), random_state=42)
    sample = pd.concat([gig_sample, shop_sample]).reset_index(drop=True)

    # Map to human-readable names
    first_names = [
        "Asha", "Rahul", "Meera", "Arjun", "Kavya", "Imran",
        "Divya", "Naveen", "Farah", "Vijay", "Sneha", "Manoj",
        "Lakshmi", "Priya", "Suresh", "Anita", "Kiran", "Deepak",
        "Nisha", "Ramesh", "Sunita", "Arun", "Geeta", "Mohan",
        "Pooja", "Sanjay", "Rekha", "Vikram", "Usha", "Ravi",
    ]
    rng.shuffle(first_names)

    rows = []
    for i, (_, r) in enumerate(sample.iterrows()):
        occupation = (
            "Gig worker"
            if r["occupation_type"] == "gig_worker"
            else "Shopkeeper"
        )
        city = r.get("city", "Kerala")
        if pd.isna(city):
            city = rng.choice(
                ["Kochi", "Thiruvananthapuram", "Thrissur", "Kozhikode"]
            )
        name = first_names[i % len(first_names)]

        rows.append({
            "id": r["applicant_id"],
            "name": f"{name}",
            "occupation": occupation,
            "city": str(city),
            "monthly_income": round(float(r.get("avg_monthly_income", 0)), 2),
            "monthly_expenses": round(float(r.get("avg_monthly_expenses", 0)), 2),
            "monthly_debt": round(
                float(r.get("self_reported_existing_monthly_debt_inr", 0)), 2
            ),
            "closing_balance": round(float(r.get("avg_closing_balance", 0)), 2),
            "transaction_regularity": round(
                float(r.get("transaction_regularity", 0.5)), 3
            ),
            "data_completeness": round(
                float(r.get("context_data_completeness", 0.5)), 3
            ),
        })

    return rows


def _fallback_profiles():
    """Fallback if data files are missing."""
    rng = random.Random(24)
    names = [
        "Asha", "Rahul", "Meera", "Arjun", "Kavya", "Imran",
        "Divya", "Naveen", "Farah", "Vijay", "Sneha", "Manoj",
    ]
    cities = [
        "Kochi", "Thiruvananthapuram", "Thrissur",
        "Kozhikode", "Visakhapatnam",
    ]
    rows = []
    for i in range(1, 31):
        gig = i % 2 == 0
        income = round(
            rng.uniform(18000, 52000) if gig else rng.uniform(30000, 85000), 2
        )
        rows.append({
            "id": f"RC-{i:04d}",
            "name": f"{rng.choice(names)} {i}",
            "occupation": "Gig worker" if gig else "Shopkeeper",
            "city": rng.choice(cities),
            "monthly_income": income,
            "monthly_expenses": round(income * rng.uniform(0.38, 0.73), 2),
            "monthly_debt": round(rng.uniform(0, income * 0.18), 2),
            "closing_balance": round(rng.uniform(500, 50000), 2),
            "transaction_regularity": round(rng.uniform(0.45, 0.99), 3),
            "data_completeness": round(rng.uniform(0.65, 1), 3),
        })
    return rows


def seed_demo_data():
    if list_applicants():
        return
    rows = _load_real_profiles()
    insert_applicants(rows)
