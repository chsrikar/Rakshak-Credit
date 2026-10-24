"""
Rakshak Credit — Phase 1 data generator
All records and repayment labels are SYNTHETIC. This is a demo dataset, not lending evidence.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
CITIES = {
    "Kerala": [("Kochi", "urban"), ("Thrissur", "semi_urban"), ("Kozhikode", "urban"),
               ("Palakkad", "semi_urban"), ("Alappuzha", "semi_urban")],
    "Andhra Pradesh": [("Visakhapatnam", "urban"), ("Vijayawada", "urban"),
                       ("Rajahmundry", "semi_urban"), ("Srikakulam", "semi_urban")],
    "Tamil Nadu": [("Coimbatore", "urban"), ("Madurai", "urban"), ("Salem", "semi_urban")],
    "Karnataka": [("Bengaluru", "urban"), ("Mysuru", "urban"), ("Hubballi", "semi_urban")],
}
PURPOSES = [
    ("working_capital", "Buy stock and maintain working capital"),
    ("vehicle_repair", "Repair my two-wheeler used for work"),
    ("equipment", "Purchase or repair work equipment"),
    ("inventory", "Purchase fast-moving shop inventory"),
    ("education", "Pay for a training or certification course"),
    ("medical", "Cover a medical expense"),
    ("home_expense", "Manage a household expense"),
    ("business_improvement", "Improve the shop or work setup"),
]
SHOP_TYPES = ["kirana_grocery", "tea_snacks", "mobile_accessories", "vegetable_fruit", "stationery"]
GIG_TYPES = ["delivery_partner", "ride_hailing_driver", "freelance_service", "field_sales", "home_service"]

def money(x):
    return round(float(max(0, x)), 2)

def make_profiles(n, rng):
    rows, gig_rows, shop_rows = [], [], []
    for i in range(n):
        occupation = str(rng.choice(["gig_worker", "small_shopkeeper"], p=[0.56, 0.44]))
        state = str(rng.choice(list(CITIES.keys())))
        city, zone = CITIES[state][int(rng.integers(0, len(CITIES[state])))]
        purpose, purpose_text = PURPOSES[int(rng.integers(0, len(PURPOSES)))]
        requested = int(rng.choice([10000, 15000, 20000, 25000, 30000, 40000, 50000, 75000]))
        tenure = int(rng.choice([3, 6, 9, 12, 18, 24]))
        if occupation == "gig_worker":
            work_type = str(rng.choice(GIG_TYPES))
            months_working = int(rng.integers(3, 97))
            workdays = int(rng.integers(12, 29))
            daily_earnings = float(rng.lognormal(np.log(1250), 0.28))
            gross_monthly = daily_earnings * workdays
            work_cost_ratio = float(rng.uniform(.12, .32))
            work_costs = gross_monthly * work_cost_ratio
            household = float(rng.uniform(6500, 15500))
            debt = float(rng.choice([0, 0, 0, 1000, 2000, 3500, 5000]))
            sources = int(rng.choice([1, 2, 3], p=[.52, .35, .13]))
            gig_rows.append({
                "applicant_id": f"RC-{i+1:05d}", "work_type": work_type,
                "platform_or_employer": str(rng.choice(["platform_A", "platform_B", "local_clients", "multiple_sources"])),
                "months_in_occupation": months_working, "workdays_per_month": workdays,
                "typical_daily_gross_earnings_inr": money(daily_earnings),
                "typical_weekly_gross_earnings_inr": money(daily_earnings * 6),
                "monthly_work_costs_inr": money(work_costs),
                "monthly_household_essential_expenses_inr": money(household),
                "existing_monthly_debt_payment_inr": money(debt),
                "income_source_count_self_reported": sources,
                "income_range_self_reported": str(rng.choice(["under_15000", "15000_25000", "25000_40000", "over_40000"],
                    p=[.15, .35, .35, .15])),
                "earnings_evidence_type": str(rng.choice(["platform_statement", "bank_transaction_pattern", "self_reported"], p=[.45,.35,.20])),
            })
            base_income = max(8000, gross_monthly - work_costs)
            business_age = None
        else:
            shop_type = str(rng.choice(SHOP_TYPES))
            business_age = int(rng.integers(6, 181))
            owned_rented = str(rng.choice(["rented", "owned"], p=[.76,.24]))
            rent = float(rng.uniform(3500, 15000)) if owned_rented == "rented" else 0.0
            customers = int(rng.integers(18, 145))
            avg_spend = float(rng.uniform(70, 360))
            days_open = int(rng.integers(24, 31))
            daily_sales = customers * avg_spend * float(rng.uniform(.70, 1.18))
            sales_month = daily_sales * days_open
            inventory_ratio = float(rng.uniform(.48, .72))
            operating_costs = sales_month * float(rng.uniform(.07, .17)) + rent
            supplier_dues = float(rng.choice([0, 0, 5000, 10000, 20000, 30000]))
            debt = float(rng.choice([0, 0, 0, 1500, 3000, 5000, 8000]))
            household = float(rng.uniform(8000, 18000))
            cashless = float(rng.uniform(.15, .85))
            shop_rows.append({
                "applicant_id": f"RC-{i+1:05d}", "business_type": shop_type,
                "business_age_months": business_age, "premises_type": owned_rented,
                "monthly_rent_inr": money(rent), "estimated_daily_customer_count": customers,
                "estimated_average_bill_inr": money(avg_spend), "estimated_days_open_per_month": days_open,
                "estimated_daily_sales_inr": money(daily_sales), "estimated_monthly_sales_inr": money(sales_month),
                "estimated_monthly_inventory_purchases_inr": money(sales_month * inventory_ratio),
                "estimated_monthly_operating_costs_inr": money(operating_costs),