"""
pipeline/export.py

Generates and exports all CSVs to the outputs/ directory.
Runs after train.py — reads saved models + raw data.
"""

import os
import pickle
import dill
import pandas as pd
import numpy as np
from datetime import datetime

# Paths 
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(ROOT, "Sample - Superstore.csv")
MODEL_DIR  = os.path.join(ROOT, "models")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_models():
    print("\n Loading saved models...")
    models = {}
    for name in ["kmeans_model.pkl", "rfm_scaler.pkl", "bgf_model.pkl"]:
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = dill.load(f)
            print(f"    {name}")
        else:
            print(f"    {name} missing — run train.py first")
    ggf_path = os.path.join(MODEL_DIR, "ggf_model.pkl")
    if os.path.exists(ggf_path):
        with open(ggf_path, "rb") as f:
            models["ggf_model.pkl"] = dill.load(f)
        print(f"    ggf_model.pkl")
    return models

def load_data():
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Sales"]      = pd.to_numeric(df["Sales"], errors="coerce")
    df = df[df["Sales"] > 0].dropna(subset=["Customer ID","Order Date","Sales"])
    df.drop_duplicates(inplace=True)
    return df

def export_customer_master(df, models):
    print("\n👥 Exporting customer master...")
    from lifetimes.utils import summary_data_from_transaction_data

    ref = df["Order Date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("Customer ID").agg(
        Recency   = ("Order Date", lambda x: (ref - x.max()).days),
        Frequency = ("Order ID",   "nunique"),
        Monetary  = ("Sales",      "sum")
    ).reset_index()

    # RFM scoring
    rfm_log = rfm[["Recency","Frequency","Monetary"]].copy()
    rfm_log["Frequency"] = np.log1p(rfm_log["Frequency"])
    rfm_log["Monetary"]  = np.log1p(rfm_log["Monetary"])
    scaler     = models["rfm_scaler.pkl"]
    kmeans     = models["kmeans_model.pkl"]
    rfm_scaled = scaler.transform(rfm_log)
    rfm["Cluster"] = kmeans.predict(rfm_scaled)

    # Auto-label
    cs = rfm.groupby("Cluster")[["Recency","Frequency","Monetary"]].mean()
    mr = cs["Monetary"].rank(ascending=False)
    rr = cs["Recency"].rank(ascending=True)
    fr = cs["Frequency"].rank(ascending=False)
    k  = len(cs)

    def label(c):
        if mr[c]==1 and rr[c]<=2:   return "Champions"
        elif mr[c]<=2 and fr[c]<=2: return "Loyal Customers"
        elif rr[c]==1:               return "Recent Customers"
        elif rr[c]>=k-1:            return "At Risk"
        else:                        return "Needs Attention"

    rfm["Segment"] = rfm["Cluster"].map({c: label(c) for c in range(k)})

    # LTV
    bgf = models.get("bgf_model.pkl")
    ggf = models.get("ggf_model.pkl")

    if bgf and ggf:
        obs_end = df["Order Date"].max()
        ltv_sum = summary_data_from_transaction_data(
            df,
            customer_id_col        = "Customer ID",
            datetime_col           = "Order Date",
            monetary_value_col     = "Sales",
            observation_period_end = obs_end,
            freq = "D"
        )
        ltv_sum = ltv_sum[ltv_sum["frequency"] > 0].copy()
        ltv_sum["LTV_12M"] = ggf.customer_lifetime_value(
            bgf,
            ltv_sum["frequency"], ltv_sum["recency"],
            ltv_sum["T"], ltv_sum["monetary_value"],
            time=12, discount_rate=0.01, freq="D"
        )
        ltv_sum["prob_alive"] = bgf.conditional_probability_alive(
            ltv_sum["frequency"], ltv_sum["recency"], ltv_sum["T"])
        for days in [90, 180, 365]:
            ltv_sum[f"pred_purchases_{days}d"] = bgf.conditional_expected_number_of_purchases_up_to_time(
                days, ltv_sum["frequency"], ltv_sum["recency"], ltv_sum["T"])

        ltv_sum = ltv_sum.reset_index()
        rfm = rfm.merge(
            ltv_sum[["Customer ID","LTV_12M","prob_alive",
                      "pred_purchases_90d","pred_purchases_180d","pred_purchases_365d"]],
            on="Customer ID", how="left"
        )

    out = os.path.join(OUTPUT_DIR, "superstore_customer_master.csv")
    rfm.to_csv(out, index=False)
    print(f"    superstore_customer_master.csv ({len(rfm):,} rows)")
    return rfm

def export_eda(df):
    print("\n Exporting EDA data...")
    df["Year"]       = df["Order Date"].dt.year
    df["Month_Num"]  = df["Order Date"].dt.month
    df["Month_Name"] = df["Order Date"].dt.strftime("%b")

    master = df.groupby(
        ["Year","Month_Num","Month_Name","Category","Region","Segment"]
    ).agg(
        Total_Sales      = ("Sales",    "sum"),
        Total_Orders     = ("Order ID", "nunique"),
        Total_Customers  = ("Customer ID", "nunique"),
        Avg_Order_Value  = ("Sales",    "mean"),
        Total_Profit     = ("Profit",   "sum") if "Profit" in df.columns else ("Sales", "count")
    ).reset_index()

    kpi = pd.DataFrame([{
        "Metric": "Total Revenue",    "Value": round(df["Sales"].sum(), 2)
    },{
        "Metric": "Unique Customers", "Value": df["Customer ID"].nunique()
    },{
        "Metric": "Total Orders",     "Value": df["Order ID"].nunique()
    },{
        "Metric": "Avg Order Value",  "Value": round(df.groupby("Order ID")["Sales"].sum().mean(), 2)
    }])

    master.to_csv(os.path.join(OUTPUT_DIR, "master_eda.csv"), index=False)
    kpi.to_csv(os.path.join(OUTPUT_DIR, "kpi_summary.csv"), index=False)
    print(f"    master_eda.csv ({len(master):,} rows)")
    print(f"    kpi_summary.csv (4 KPIs)")

def export_cohort(df):
    print("\n🔄 Exporting cohort retention...")
    df = df.copy()
    df["CohortMonth"] = df.groupby("Customer ID")["Order Date"].transform("min").dt.to_period("M")
    df["OrderMonth"]  = df["Order Date"].dt.to_period("M")
    df["CohortIndex"] = (df["OrderMonth"] - df["CohortMonth"]).apply(lambda x: x.n)

    cohort = df.groupby(["CohortMonth","CohortIndex"])["Customer ID"].nunique().reset_index()
    cohort.columns = ["CohortMonth","CohortIndex","Customers"]
    cohort_pivot   = cohort.pivot(index="CohortMonth", columns="CohortIndex", values="Customers")
    cohort_pct     = cohort_pivot.divide(cohort_pivot[0], axis=0).round(4) * 100

    cohort_pct.to_csv(os.path.join(OUTPUT_DIR, "cohort_retention.csv"))
    print(f"    cohort_retention.csv ({len(cohort_pct)} cohorts)")

def run(models=None):
    print("=" * 55)
    print("  SUPERSTORE PIPELINE — EXPORT")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 55)

    df = load_data()
    if models is None:
        models = load_models()

    export_customer_master(df, models)
    export_eda(df)
    export_cohort(df)

    print("\n" + "=" * 55)
    print("  EXPORT COMPLETE ")
    print(f"  All files saved to /outputs")
    print("=" * 55)

if __name__ == "__main__":
    run()