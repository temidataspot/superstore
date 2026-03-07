"""
pipeline/evaluate.py

Validates model outputs and writes model_report.json
to the outputs/ directory.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

#  Paths
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(ROOT, "models")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
DATA_PATH  = os.path.join(ROOT, "Sample - Superstore.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_meta():
    path = os.path.join(MODEL_DIR, "train_meta.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError("train_meta.pkl not found — run train.py first")
    with open(path, "rb") as f:
        return pickle.load(f)

def validate_models():
    print("\n Validating saved model artifacts...")
    expected = ["kmeans_model.pkl", "rfm_scaler.pkl", "bgf_model.pkl"]
    results  = {}
    for fname in expected:
        path   = os.path.join(MODEL_DIR, fname)
        exists = os.path.exists(path)
        size   = round(os.path.getsize(path) / 1024, 2) if exists else 0
        results[fname] = {"exists": exists, "size_kb": size}
        status = f"  {size} KB" if exists else "  MISSING"
        print(f"   {fname:<25} {status}")

    optional = "ggf_model.pkl"
    path     = os.path.join(MODEL_DIR, optional)
    if os.path.exists(path):
        size = round(os.path.getsize(path) / 1024, 2)
        print(f"   {optional:<25}   {size} KB (optional)")
        results[optional] = {"exists": True, "size_kb": size}
    else:
        print(f"   {optional:<25}   Not present (correlation threshold not met)")

    return results

def compute_eda_stats():
    print("\n Computing EDA statistics...")
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Sales"]      = pd.to_numeric(df["Sales"], errors="coerce")
    df = df[df["Sales"] > 0].dropna(subset=["Customer ID","Order Date","Sales"])

    stats = {
        "total_revenue":    round(float(df["Sales"].sum()), 2),
        "unique_customers": int(df["Customer ID"].nunique()),
        "total_orders":     int(df["Order ID"].nunique()),
        "avg_order_value":  round(float(df.groupby("Order ID")["Sales"].sum().mean()), 2),
        "date_range": {
            "start": df["Order Date"].min().strftime("%Y-%m-%d"),
            "end":   df["Order Date"].max().strftime("%Y-%m-%d"),
        },
        "top_category": df.groupby("Category")["Sales"].sum().idxmax(),
        "top_region":   df.groupby("Region")["Sales"].sum().idxmax(),
    }

    print(f"   Revenue:   ${stats['total_revenue']:,.2f}")
    print(f"   Customers: {stats['unique_customers']:,}")
    print(f"   Orders:    {stats['total_orders']:,}")
    print(f"   AOV:       ${stats['avg_order_value']:,.2f}")
    return stats

def sanity_checks(meta, eda):
    print("\n Running sanity checks...")
    checks = {}

    # LTV sanity
    if meta.get("avg_ltv"):
        ltv_ok = 100 < meta["avg_ltv"] < 50000
        checks["ltv_in_range"] = ltv_ok
        print(f"   LTV in range ($100–$50K):  {'' if ltv_ok else '❌'}  avg=${meta['avg_ltv']}")

    # Revenue sanity
    rev_ok = eda["total_revenue"] > 0
    checks["revenue_positive"] = rev_ok
    print(f"   Revenue positive:          {'' if rev_ok else '❌'}")

    # Customer count sanity
    cust_ok = eda["unique_customers"] >= 10
    checks["min_customers"] = cust_ok
    print(f"   Min 10 customers:          {'' if cust_ok else '❌'}  got {eda['unique_customers']}")

    # Model files exist
    models_ok = os.path.exists(os.path.join(MODEL_DIR, "bgf_model.pkl"))
    checks["bgf_exists"] = models_ok
    print(f"   BG/NBD model saved:        {'' if models_ok else '❌'}")

    all_passed = all(checks.values())
    print(f"\n   {'All checks passed 🎉' if all_passed else 'Some checks failed ⚠️'}")
    return checks

def write_report(meta, eda, model_files, checks):
    report = {
        "generated_at":  datetime.utcnow().isoformat(),
        "trained_at":    meta.get("trained_at"),
        "eda": eda,
        "ltv": {
            "customers_modelled": meta.get("customers_ltv"),
            "avg_ltv_12m":        meta.get("avg_ltv"),
            "median_ltv_12m":     meta.get("median_ltv"),
            "top10_ltv_12m":      meta.get("top10_ltv"),
        },
        "rfm": {
            "optimal_k": meta.get("best_k"),
        },
        "model_artifacts": model_files,
        "sanity_checks":   checks,
        "all_checks_passed": all(checks.values()),
    }

    out_path = os.path.join(OUTPUT_DIR, "model_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n Report saved → outputs/model_report.json")
    return report

def run(meta=None):
    print("=" * 55)
    print("  SUPERSTORE PIPELINE — EVALUATION")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 55)

    if meta is None:
        meta = load_meta()

    model_files = validate_models()
    eda         = compute_eda_stats()
    checks      = sanity_checks(meta, eda)
    report      = write_report(meta, eda, model_files, checks)

    print("\n" + "=" * 55)
    print("  EVALUATION COMPLETE ")
    print("=" * 55)
    return report

if __name__ == "__main__":
    run()