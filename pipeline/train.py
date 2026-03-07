"""
pipeline/train.py

Trains BG/NBD, Gamma-Gamma and K-Means models
on the Superstore transaction data.
Saves fitted model artifacts to /models directory.
"""

import os
import pickle
import dill
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

# Paths 
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "Sample - Superstore.csv")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    print(" Loading data...")
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Sales"]      = pd.to_numeric(df["Sales"], errors="coerce")
    df = df[df["Sales"] > 0].dropna(subset=["Customer ID","Order Date","Sales"])
    df.drop_duplicates(inplace=True)
    print(f"   Loaded {len(df):,} rows · {df['Customer ID'].nunique():,} customers")
    return df

def train_rfm(df):
    print("\n Training RFM K-Means model...")
    ref = df["Order Date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("Customer ID").agg(
        Recency   = ("Order Date", lambda x: (ref - x.max()).days),
        Frequency = ("Order ID",   "nunique"),
        Monetary  = ("Sales",      "sum")
    ).reset_index()

    rfm_log = rfm[["Recency","Frequency","Monetary"]].copy()
    rfm_log["Frequency"] = np.log1p(rfm_log["Frequency"])
    rfm_log["Monetary"]  = np.log1p(rfm_log["Monetary"])
    scaler     = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    K_range = range(2, min(8, len(rfm)//20 + 2))
    sils    = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        sils.append(silhouette_score(rfm_scaled, km.fit_predict(rfm_scaled)))
    best_k = list(K_range)[np.argmax(sils)]
    print(f"   Optimal K = {best_k} (silhouette = {max(sils):.4f})")

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    cs = rfm.groupby("Cluster")[["Recency","Frequency","Monetary"]].mean()
    mr = cs["Monetary"].rank(ascending=False)
    fr = cs["Frequency"].rank(ascending=False)
    rr = cs["Recency"].rank(ascending=True)

    def label(c):
        if mr[c]==1 and rr[c]<=2:       return "Champions"
        elif mr[c]<=2 and fr[c]<=2:     return "Loyal Customers"
        elif rr[c]==1:                   return "Recent Customers"
        elif rr[c]>=best_k-1:           return "At Risk"
        else:                            return "Needs Attention"

    rfm["Segment"] = rfm["Cluster"].map({c: label(c) for c in range(best_k)})
    print(f"   Segments: {rfm['Segment'].value_counts().to_dict()}")

    with open(os.path.join(MODEL_DIR, "kmeans_model.pkl"), "wb") as f:
        pickle.dump(kmeans, f)
    with open(os.path.join(MODEL_DIR, "rfm_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print("   K-Means + Scaler saved to /models")
    return rfm, kmeans, scaler, best_k

def train_ltv(df):
    print("\n Training BG/NBD + Gamma-Gamma LTV model...")
    obs_end = df["Order Date"].max()

    ltv_sum = summary_data_from_transaction_data(
        df,
        customer_id_col        = "Customer ID",
        datetime_col           = "Order Date",
        monetary_value_col     = "Sales",
        observation_period_end = obs_end,
        freq = "D"
    )
    ltv_sum = ltv_sum[ltv_sum["frequency"] > 0]
    print(f"   Modelling {len(ltv_sum):,} repeat purchasers")

    bgf = None
    for p in [0.01, 0.1, 0.5, 1.0, 2.0]:
        try:
            bgf = BetaGeoFitter(penalizer_coef=p)
            bgf.fit(ltv_sum["frequency"], ltv_sum["recency"], ltv_sum["T"])
            print(f"   BG/NBD converged  (penalizer={p})")
            break
        except Exception:
            bgf = None

    if bgf is None:
        print("    BG/NBD failed to converge — skipping LTV")
        return ltv_sum, None, None

    for days in [90, 180, 365]:
        ltv_sum[f"pred_purchases_{days}d"] = bgf.conditional_expected_number_of_purchases_up_to_time(
            days, ltv_sum["frequency"], ltv_sum["recency"], ltv_sum["T"])
    ltv_sum["prob_alive"] = bgf.conditional_probability_alive(
        ltv_sum["frequency"], ltv_sum["recency"], ltv_sum["T"])

    ggf  = None
    corr = abs(ltv_sum[["frequency","monetary_value"]].corr().iloc[0,1])
    print(f"   Frequency–Monetary correlation: {corr:.4f}")

    if corr < 0.3:
        for p in [0.001, 0.01, 0.1, 0.5]:
            try:
                ggf = GammaGammaFitter(penalizer_coef=p)
                ggf.fit(ltv_sum["frequency"], ltv_sum["monetary_value"])
                ltv_sum["LTV_12M"] = ggf.customer_lifetime_value(
                    bgf,
                    ltv_sum["frequency"], ltv_sum["recency"],
                    ltv_sum["T"], ltv_sum["monetary_value"],
                    time=12, discount_rate=0.01, freq="D"
                )
                print(f"   Gamma-Gamma fitted  (penalizer={p})")
                break
            except Exception:
                ggf = None
    else:
        print("     Correlation too high — Gamma-Gamma skipped")

    with open(os.path.join(MODEL_DIR, "bgf_model.pkl"), "wb") as f:
        dill.dump(bgf, f)
    if ggf:
        with open(os.path.join(MODEL_DIR, "ggf_model.pkl"), "wb") as f:
            dill.dump(ggf, f)

    print("    BG/NBD + Gamma-Gamma saved to /models")
    return ltv_sum, bgf, ggf

def save_metadata(best_k, ltv_sum):
    meta = {
        "trained_at":    datetime.utcnow().isoformat(),
        "best_k":        best_k,
        "customers_ltv": int(len(ltv_sum)),
        "avg_ltv":       float(round(ltv_sum["LTV_12M"].mean(), 2)) if "LTV_12M" in ltv_sum.columns else None,
        "median_ltv":    float(round(ltv_sum["LTV_12M"].median(), 2)) if "LTV_12M" in ltv_sum.columns else None,
        "top10_ltv":     float(round(ltv_sum["LTV_12M"].quantile(0.9), 2)) if "LTV_12M" in ltv_sum.columns else None,
    }
    with open(os.path.join(MODEL_DIR, "train_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"\n Metadata saved: {meta}")

def run():
    print("=" * 55)
    print("  SUPERSTORE PIPELINE — TRAINING")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 55)

    df                          = load_data()
    rfm, kmeans, scaler, best_k = train_rfm(df)
    ltv_sum, bgf, ggf           = train_ltv(df)
    save_metadata(best_k, ltv_sum)

    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE ")
    print("=" * 55)

    return df, rfm, ltv_sum, bgf, ggf, kmeans, scaler

if __name__ == "__main__":
    run()