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
                "supplier_outstanding_inr": money(supplier_dues),
                "estimated_cashless_sales_ratio": round(cashless, 3),
                "monthly_household_essential_expenses_inr": money(household),
                "existing_monthly_debt_payment_inr": money(debt),
                "sales_evidence_type": str(rng.choice(["sales_ledger", "bank_transaction_pattern", "self_reported"], p=[.30,.35,.35])),
            })
            # approximate retained income after stock/operating costs
            base_income = max(8000, sales_month * (1-inventory_ratio) - operating_costs)
            work_type = None
            workdays = days_open
            daily_earnings = daily_sales
            work_costs = sales_month * inventory_ratio + operating_costs
            sources = int(rng.choice([1,2,3], p=[.20,.45,.35]))

        # Common application fields; free text is stored for UI/analyst review, not directly fed to model.
        rows.append({
            "applicant_id": f"RC-{i+1:05d}",
            "application_date": "2026-10-23",
            "occupation_type": occupation,
            "state": state, "city": city, "work_zone_type": zone,
            "loan_purpose_category": purpose, "loan_purpose_text": purpose_text,
            "requested_loan_amount_inr": requested, "requested_tenure_months": tenure,
            "self_reported_monthly_income_inr": money(base_income),
            "self_reported_monthly_essential_expenses_inr": money(household if occupation=="small_shopkeeper" else household),
            "self_reported_existing_monthly_debt_inr": money(debt),
            "profile_data_source": "synthetic_self_reported",
            "consent_status": "demo_consent_granted",
            "identity_document_collected": False,
            "sensitive_attributes_collected": False,
        })
    return pd.DataFrame(rows), pd.DataFrame(gig_rows), pd.DataFrame(shop_rows)

def make_transactions(applicants, gig, shops, rng):
    app_idx = applicants.set_index("applicant_id")
    gig_idx = gig.set_index("applicant_id") if len(gig) else pd.DataFrame()
    shop_idx = shops.set_index("applicant_id") if len(shops) else pd.DataFrame()
    rows = []
    start = pd.Timestamp("2026-04-01")
    categories = ["income", "business_income", "food", "rent", "fuel", "utilities", "inventory",
                  "transport", "health", "shopping", "transfer", "supplier_payment", "equipment"]
    for aid, a in app_idx.iterrows():
        is_shop = a["occupation_type"] == "small_shopkeeper"
        if is_shop:
            p = shop_idx.loc[aid]
            monthly_base = float(p["estimated_monthly_sales_inr"])
            expense_base = float(p["estimated_monthly_inventory_purchases_inr"] + p["estimated_monthly_operating_costs_inr"])
            income_cat = "business_income"
            payout_n = int(rng.integers(18, 35))
            source_name = "merchant_receipt"
        else:
            p = gig_idx.loc[aid]
            monthly_base = float(p["typical_daily_gross_earnings_inr"] * p["workdays_per_month"])
            expense_base = float(p["monthly_work_costs_inr"] + p["monthly_household_essential_expenses_inr"])
            income_cat = "income"
            payout_n = int(rng.integers(5, 13))
            source_name = "gig_payout"
        opening = float(rng.uniform(1200, 18000))
        for m in range(6):
            month_start = start + pd.DateOffset(months=m)
            last_day = (month_start + pd.offsets.MonthEnd(0)).day
            vol = float(rng.uniform(.08,.28) if is_shop else rng.uniform(.12,.48))
            scale = float(rng.lognormal(0, vol))
            monthly_income = monthly_base * scale
            # Simulate several receipt events; shops have more frequent digital receipts.
            days = rng.integers(1, last_day+1, size=payout_n)
            amounts = rng.dirichlet(np.ones(payout_n)*2.2) * monthly_income
            events = []
            for j, (day, amount) in enumerate(zip(days, amounts)):
                events.append((month_start + pd.Timedelta(days=int(day)-1), float(max(80,amount)),
                               "credit", income_cat, f"{source_name}_{j % (3 if is_shop else int(p.get('income_source_count_self_reported', 2)) + 1)}"))
            exp_n = int(rng.integers(22, 39))
            exp_days = rng.integers(1, last_day+1, size=exp_n)
            if is_shop:
                exp_cats = rng.choice(["inventory","supplier_payment","rent","utilities","transport","food","transfer","equipment"],
                                      size=exp_n, p=[.35,.16,.10,.08,.08,.08,.08,.07])
            else:
                exp_cats = rng.choice(["fuel","food","rent","utilities","transport","health","shopping","transfer","equipment"],
                                      size=exp_n, p=[.22,.22,.10,.07,.13,.04,.09,.07,.06])
            weights = rng.dirichlet(np.ones(exp_n)*1.8)
            expense_total = monthly_income * (float(rng.uniform(.66,.91)) if is_shop else float(rng.uniform(.58,.86)))
            for day, cat, w in zip(exp_days, exp_cats, weights):
                events.append((month_start + pd.Timedelta(days=int(day)-1), float(max(35, expense_total*float(w))),
                               "debit", str(cat), str(cat)))
            # Small transfers are included but deliberately not classified as earned income.
            for _ in range(int(rng.poisson(1.5))):
                day = int(rng.integers(1,last_day+1))
                events.append((month_start + pd.Timedelta(days=day-1), float(rng.uniform(100,1500)),
                               "credit", "transfer", "own_or_peer_transfer"))
            events.sort(key=lambda x: x[0])
            for dt, amount, direction, cat, desc in events:
                signed = amount if direction == "credit" else -amount
                opening += signed
                rows.append((aid, dt.date().isoformat(), round(amount,2), direction, cat, desc,
                             round(opening,2), "synthetic_bank_or_upi_feed", True))
    tx = pd.DataFrame(rows, columns=["applicant_id","transaction_date","amount_inr","direction","category",
                                    "description","running_balance_inr","source_type","consented_demo_data"])
    return tx.sort_values(["applicant_id","transaction_date"], kind="stable").reset_index(drop=True)

def build_features(applicants, gig, shops, tx, rng):
    t = tx.copy()
    t["transaction_date"] = pd.to_datetime(t["transaction_date"])
    t["month"] = t["transaction_date"].dt.to_period("M").astype(str)
    t["earned_income"] = np.where(t["category"].isin(["income","business_income"]), t["amount_inr"], 0.0)
    t["debit"] = np.where(t["direction"].eq("debit"), t["amount_inr"], 0.0)
    monthly = t.groupby(["applicant_id","month"], as_index=False).agg(
        monthly_earned_income=("earned_income","sum"),
        monthly_expenses=("debit","sum"),
        closing_balance=("running_balance_inr","last"),
        transaction_count=("amount_inr","size"),
    )
    monthly["net_cash_flow"] = monthly["monthly_earned_income"] - monthly["monthly_expenses"]
    g = monthly.groupby("applicant_id")
    f = g.agg(
        avg_monthly_income=("monthly_earned_income","mean"),
        median_monthly_income=("monthly_earned_income","median"),
        std_monthly_income=("monthly_earned_income","std"),
        positive_income_months=("monthly_earned_income",lambda s:int((s>0).sum())),
        avg_monthly_expenses=("monthly_expenses","mean"),
        avg_closing_balance=("closing_balance","mean"),
        negative_cash_flow_months_ratio=("net_cash_flow",lambda s:float((s<0).mean())),
        net_cash_flow_6m=("net_cash_flow","sum"),
        avg_monthly_transactions=("transaction_count","mean"),
    )
    f["income_cv"] = (f["std_monthly_income"] / f["avg_monthly_income"].replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0)
    f["positive_income_months_ratio"] = f["positive_income_months"] / 6
    def trend(s):
        x = np.arange(len(s), dtype=float)
        return float(np.polyfit(x, s.to_numpy(), 1)[0]) if len(s)>1 else 0.0
    f["income_trend"] = g["monthly_earned_income"].apply(trend)
    t["transaction_day"] = t["transaction_date"].dt.date
    daily = t.groupby(["applicant_id", "transaction_day"], as_index=False).agg(
        daily_balance=("running_balance_inr", "last"), daily_income=("earned_income", "sum"))
    low = daily.groupby("applicant_id")["daily_balance"].apply(lambda s: float((s < 2000).mean()))
    f["low_balance_days_ratio"] = low
    earned = t[t["earned_income"] > 0]
    conc = earned.groupby("applicant_id")["description"].apply(lambda s: float(s.value_counts(normalize=True).max()) if len(s) else 1.0)
    f["income_source_concentration"] = conc
    regularity = {}
    for aid, group in earned.groupby("applicant_id"):
        gaps_days = group["transaction_date"].sort_values().diff().dt.days.dropna()
        gap_std = float(gaps_days.std()) if len(gaps_days) > 1 and pd.notna(gaps_days.std()) else 0.0
        regularity[aid] = float(1.0 / (1.0 + gap_std / 30.0)) if len(gaps_days) else 0.0
    f["transaction_regularity"] = pd.Series(regularity)
    essential_cats = ["food","rent","fuel","utilities","health","transport"]
    essential = t[t["category"].isin(essential_cats)].groupby("applicant_id")["amount_inr"].sum()
    debits = t[t["direction"].eq("debit")].groupby("applicant_id")["amount_inr"].sum()
    f["essential_expense_ratio"] = (essential/debits).fillna(0)
    f["savings_ratio"] = ((f["avg_monthly_income"]-f["avg_monthly_expenses"])/f["avg_monthly_income"].replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0)
    f["cash_flow_coverage_ratio"] = f["avg_monthly_income"]/(f["avg_monthly_expenses"]+1)
    # Join relevant numeric context. Text fields, city, loan purpose and IDs are retained elsewhere only.
    app = applicants.set_index("applicant_id")
    f["occupation_gig_worker"] = (app["occupation_type"]=="gig_worker").astype(int)
    f["requested_loan_amount_inr"] = app["requested_loan_amount_inr"]
    f["requested_tenure_months"] = app["requested_tenure_months"]
    f["self_reported_monthly_essential_expenses_inr"] = app["self_reported_monthly_essential_expenses_inr"]
    f["self_reported_existing_monthly_debt_inr"] = app["self_reported_existing_monthly_debt_inr"]
    f["context_data_completeness"] = 0.0
    gig_idx = gig.set_index("applicant_id") if len(gig) else pd.DataFrame()
    shop_idx = shops.set_index("applicant_id") if len(shops) else pd.DataFrame()
    for aid in f.index:
        if aid in gig_idx.index:
            p = gig_idx.loc[aid]
            f.loc[aid,"months_in_occupation"] = p["months_in_occupation"]
            f.loc[aid,"workdays_per_month"] = p["workdays_per_month"]
            f.loc[aid,"monthly_work_costs_inr"] = p["monthly_work_costs_inr"]
            f.loc[aid,"occupation_numeric_1"] = p["typical_daily_gross_earnings_inr"]
            f.loc[aid,"occupation_numeric_2"] = p["income_source_count_self_reported"]
            f.loc[aid,"context_data_completeness"] = .8 if p["earnings_evidence_type"] != "self_reported" else .45
        elif aid in shop_idx.index:
            p = shop_idx.loc[aid]
            f.loc[aid,"months_in_occupation"] = p["business_age_months"]
            f.loc[aid,"workdays_per_month"] = p["estimated_days_open_per_month"]
            f.loc[aid,"monthly_work_costs_inr"] = p["estimated_monthly_inventory_purchases_inr"] + p["estimated_monthly_operating_costs_inr"]
            f.loc[aid,"occupation_numeric_1"] = p["estimated_daily_customer_count"]
            f.loc[aid,"occupation_numeric_2"] = p["estimated_average_bill_inr"]
            f.loc[aid,"context_data_completeness"] = .8 if p["sales_evidence_type"] != "self_reported" else .45
    f = f.replace([np.inf,-np.inf],np.nan).fillna(0).reset_index()
    # Toy target mechanism only for demonstrating the pipeline. Do NOT use probability as a feature.
    expense_burden = (f["avg_monthly_expenses"] / f["avg_monthly_income"].clip(lower=1)).clip(0,2)
    low_buffer = (f["avg_closing_balance"] < 3000).astype(float)
    volatility = f["income_cv"].clip(0,1.5)
    rng_noise = rng.normal(0, .65, len(f))
    logit = -2.3 + 1.65*volatility + 1.65*expense_burden + .65*f["negative_cash_flow_months_ratio"] + .45*low_buffer - .000003*f["avg_monthly_income"].clip(lower=0) + rng_noise
    prob = 1/(1+np.exp(-np.clip(logit,-8,8)))
    f["synthetic_missed_payment"] = (rng.random(len(f)) < prob).astype("int8")
    f["synthetic_target_probability_for_documentation_only"] = prob.round(5)
    return f

MODEL_FEATURES = [
    "avg_monthly_income","median_monthly_income","income_cv","positive_income_months_ratio",
    "income_trend","avg_monthly_expenses","essential_expense_ratio","savings_ratio",
    "avg_closing_balance","low_balance_days_ratio","net_cash_flow_6m",
    "negative_cash_flow_months_ratio","income_source_concentration","transaction_regularity",
    "cash_flow_coverage_ratio","occupation_gig_worker","requested_loan_amount_inr",
    "requested_tenure_months","self_reported_monthly_essential_expenses_inr",
    "self_reported_existing_monthly_debt_inr","months_in_occupation","workdays_per_month",
    "monthly_work_costs_inr","occupation_numeric_1","occupation_numeric_2","context_data_completeness"
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--applicants", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()
    if args.applicants < 100:
        raise SystemExit("Use at least 100 applicants so train/validation/test splits are meaningful.")
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    applicants, gig, shops = make_profiles(args.applicants, rng)
    tx = make_transactions(applicants, gig, shops, rng)
    features = build_features(applicants, gig, shops, tx, rng)
    for c in MODEL_FEATURES:
        if c not in features: features[c] = 0.0
    features[MODEL_FEATURES] = features[MODEL_FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0).astype("float32")
    # Applicant-level splits prevent a single applicant's rows leaking across partitions.
    ids_train, ids_temp = train_test_split(features["applicant_id"], test_size=.30, random_state=args.seed,
                                           stratify=features["synthetic_missed_payment"])
    temp_labels = features.set_index("applicant_id").loc[ids_temp, "synthetic_missed_payment"]
    ids_val, ids_test = train_test_split(ids_temp, test_size=.50, random_state=args.seed, stratify=temp_labels)
    split_by_id = {**{x:"train" for x in ids_train}, **{x:"validation" for x in ids_val}, **{x:"test" for x in ids_test}}
    features["split"] = features["applicant_id"].map(split_by_id)
    # Expose a clean ML table without the target-generation probability, raw PII, free text, or location.
    model_table = features[["applicant_id"] + MODEL_FEATURES + ["synthetic_missed_payment","split"]].copy()
    # Add split file outputs for convenience.
    applicants.to_csv(args.out/"applicants.csv", index=False)
    gig.to_csv(args.out/"gig_worker_profiles.csv", index=False)
    shops.to_csv(args.out/"shopkeeper_profiles.csv", index=False)
    tx.to_csv(args.out/"transactions.csv.gz", index=False, compression="gzip")
    features.to_csv(args.out/"engineered_features_with_audit.csv", index=False)
    model_table.to_csv(args.out/"model_ready_dataset.csv", index=False)
    for split in ["train","validation","test"]:
        model_table[model_table["split"]==split].drop(columns="split").to_csv(args.out/f"{split}.csv", index=False)
    # Data dictionary and leakage guardrails.
    dictionary = {
        "synthetic_notice":"Every row and label is generated. Never represent these labels or model metrics as real-world credit performance.",
        "applicants.csv":"One row per application; shared application, requested loan, location and purpose fields.",
        "gig_worker_profiles.csv":"Gig-worker-only profile. Numeric estimates are not guaranteed verified facts.",
        "shopkeeper_profiles.csv":"Shopkeeper-only profile, including customer counts, bill size, sales, stock purchases, costs and supplier dues.",
        "transactions.csv.gz":"Synthetic consented-demo transaction feed, 6 months per applicant. Amounts and balances are INR.",
        "engineered_features_with_audit.csv":"Model features plus synthetic target and documentation-only target probability; use for analysis, not direct training as-is.",
        "model_ready_dataset.csv":"Curated numeric feature table plus target and applicant-level split. Drop applicant_id and split before model.fit.",
        "target":"synthetic_missed_payment is generated by a toy risk formula, not observed repayment outcomes.",
        "excluded_from_model":"City, state, work zone, loan purpose free text, platform/employer string, shop type, audit group and synthetic_target_probability_for_documentation_only are excluded from the starter model features.",
        "fairness":"Location and other contextual fields may proxy for socioeconomic differences. Do not add them to the model without a defensible purpose, validation and fairness review.",
        "source_quality":"Evidence/source fields are metadata. In production, independently verify what is verified and retain consent, purpose, retention and access logs."
    }
    (args.out/"DATA_DICTIONARY.json").write_text(json.dumps(dictionary, indent=2), encoding="utf-8")
    (args.out/"model_feature_list.json").write_text(json.dumps({
        "target":"synthetic_missed_payment",
        "feature_columns":MODEL_FEATURES,
        "excluded_columns":["applicant_id","split","synthetic_target_probability_for_documentation_only","city","state","work_zone_type","loan_purpose_category","loan_purpose_text","platform_or_employer","business_type","audit_group"],
        "warning":"This is a demo feature set. Validate feature availability, reliability, fairness, calibration and policy before any real use."
    }, indent=2), encoding="utf-8")
    counts = features["synthetic_missed_payment"].value_counts().to_dict()
    report = {
        "synthetic_only": True, "seed":args.seed, "applicants":len(applicants),
        "gig_workers":len(gig), "shopkeepers":len(shops), "transactions":len(tx),
        "months_of_transactions":6, "model_feature_count":len(MODEL_FEATURES),
        "target_counts":{str(k):int(v) for k,v in counts.items()},
        "split_counts":features["split"].value_counts().to_dict(),
        "model_features":MODEL_FEATURES,
        "warnings":[
            "No real repayment outcomes are present; the target is synthetic.",
            "The displayed score/limit must be labeled as an estimate, not approval or a CIBIL score.",
            "Do not feed free text or location into the starter model.",
            "The affordability calculation should be a separate policy layer from ML."
        ]
    }
    (args.out/"data_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
