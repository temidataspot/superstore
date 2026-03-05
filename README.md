# 🛒 E-Commerce Marketing Analytics
### RFM Segmentation · BG/NBD + Gamma-Gamma LTV · Cohort Retention Analysis

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-K--Means-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Lifetimes](https://img.shields.io/badge/Lifetimes-BG%2FNBD-38bdf8?style=flat)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=flat&logo=plotly&logoColor=white)

> **Live App →** [Marklytics](https://marklytic.streamlit.app/)  
> **Notebook →** [Superstore_Analytics.ipynb](https://github.com/temidataspot/superstore/blob/main/Superstore.ipynb)  
> **Dataset →** [Superstore Sales](https://github.com/temidataspot/superstore/blob/main/Sample%20-%20Superstore.csv)

---

## Project Overview

This project delivers a **complete customer intelligence pipeline** on the Superstore retail dataset — moving from raw transaction data to actionable business strategy across three industry-standard analyses:

| Analysis | Method | Output |
|---|---|---|
| **RFM Segmentation** | K-Means Clustering | Customer segments with behavioural profiles |
| **LTV Prediction** | BG/NBD + Gamma-Gamma | 12-month revenue forecast per customer |
| **Cohort Retention** | Cohort Heatmap | Monthly retention rates across 42 cohorts |

The project is deployed as a **live Streamlit app** that accepts any transaction CSV — making it dataset-agnostic and production-ready.

---

## Dataset

| Property | Value |
|---|---|
| Source | IBM Superstore Sales (Kaggle) |
| Rows | 9,994 transactions |
| Customers | 793 unique customers |
| Date Range | Jan 2014 – Dec 2017 |
| Total Revenue | $2,297,200.86 |
| Missing Values | None |
| Duplicates | None |

**Columns used:** `Customer ID` · `Order ID` · `Order Date` · `Sales`

---

##  Exploratory Data Analysis

### Key Findings

- **Peak month: November** — pre-holiday demand drives highest monthly revenue
- **YoY growth:** -2.8% (2014→2015) then recovery to +29.5% (2015→2016) and +20.4% (2016→2017)
- **Technology** is the highest-revenue category; **Furniture** the lowest despite high order values
- **Consumer segment** drives the most volume; **Corporate** has the highest avg order value

### Visuals

| Chart | Insight |
|---|---|
| Sales by Month Name | November peak, Q4 dominance |
| Sales by Year | Strong recovery and growth 2015–2017 |
| YoY Monthly Comparison | Consistent seasonal patterns across all years |
| Month × Year Heatmap | Q4 (Oct–Dec) consistently outperforms all other quarters |

---

##  RFM Segmentation — K-Means Clustering

### Why K-Means over Manual Scoring?

Traditional RFM assigns manual 1–5 scores — subjective and arbitrary. K-Means **discovers natural groupings** in the data automatically, finding the boundaries that actually exist rather than imposing artificial ones.

### Finding Optimal K

Silhouette Score analysis identified **K=2** as the natural structure in this dataset — two genuinely distinct customer populations rather than forced multi-segment splits.

### Results

| Segment | Customers | Avg Recency | Avg Orders | Avg Revenue |
|---|---|---|---|---|
| **Champions** | 576 (72.6%) | 85 days | 7.3 orders | $3,611 |
| **Loyal Customers** | 217 (27.4%) | 315 days | 3.7 orders | $1,002 |

### Interpretation

**Champions (72.6%)** purchased recently, buy frequently and spend the most — these are your most valuable customers. They spend **3.6× more** than Loyal Customers and buy **twice as often**.

**Loyal Customers (27.4%)** are historically engaged but haven't purchased in ~10 months on average — they are at risk of churning and represent a significant win-back opportunity.

---

##  LTV Prediction — BG/NBD + Gamma-Gamma

### Why This Approach?

> Standard ML regression struggles with LTV because transaction data is inherently skewed and non-linear. The **BG/NBD + Gamma-Gamma** probabilistic model is purpose-built for non-contractual customer transactions — it's the same framework used by **Amazon, Netflix and Spotify** in production.

### How It Works

```
Part 1 — BG/NBD Model
  Inputs:  Recency, Frequency, Customer Age (T)
  Output:  Expected number of future purchases

Part 2 — Gamma-Gamma Model  
  Inputs:  Frequency, Average Order Value
  Output:  Expected revenue per future purchase

Combined:
  LTV = Expected Purchases × Expected Order Value × Discount Factor
```

### Model Validation

| Check | Result | Status |
|---|---|---|
| BG/NBD convergence | Converged at penalizer=0.01 | ✅ |
| Frequency–Monetary correlation | -0.0074 (< 0.3 required) | ✅ |
| Gamma-Gamma fit | p=9.79, q=0.72, v=10.14 | ✅ |

### LTV Results

| Metric | Value |
|---|---|
| Customers modelled | 781 |
| Average predicted 12M LTV | $723.75 |
| Median predicted 12M LTV | $574.64 |
| Top 10% customer LTV | $1,450.82 |

### LTV by Segment

| Segment | Customers | Avg 12M LTV |
|---|---|---|
| **Champions** | 576 | **$859.18** |
| **Loyal Customers** | 217 | $343.23 |

Champions are predicted to generate **2.5× more revenue** over the next 12 months than Loyal Customers — directly informing where to concentrate retention and upsell investment.

### Model Limitation

`prob_alive = 100%` across all customers is expected for this dataset. The Superstore spans only 4 years — insufficient for the BG/NBD model to observe customer "death" (permanent churn). In production, denser transaction history (5+ years) significantly improves this estimate. LTV and purchase forecasts remain valid.

---

## Cohort Retention Analysis

### Setup

- **42 monthly cohorts** (Jan 2014 – Nov 2017)
- Each cohort tracked for up to 48 months
- Month 0 = 100% by definition (first purchase month)

### Key Findings

| Finding | Value |
|---|---|
| Month 1 retention | ~12% |
| Steady-state retention (Month 2–45) | 15–20% |
| Long-tenure retention (Month 40+) | ~26% |
| High-retention cohorts | Jun 2015, Feb–Mar 2016 (42–57%) |

### Interpretation

The retention curve reveals two distinct customer populations:

1. **One-time buyers (88%)** — never return after first purchase
2. **Loyal repeat buyers (12%)** — maintain consistent 15–20% monthly activity for 3+ years

The **Month 1 drop-off from 100% → 12%** is the single most important business metric. Every 1% improvement in Month 1 retention compounds significantly over the 45-month customer lifetime.

---

## 💼 Business Recommendations

### By Segment

| Segment | Priority | Action |
|---|---|---|
| **Champions** | Retain & leverage | VIP programme, referral incentives, early product access, loyalty rewards |
| **Loyal Customers** | Reactivate urgently | Re-engagement campaign within 30 days, personalised discount, "we miss you" email |

### Retention Strategy

- **Day 14 post-purchase email** — personalised product recommendations based on first order
- **30-day second purchase incentive** — customers who buy twice within 30 days have 3× higher 12M LTV
- **Segment-based CAC cap** — acquisition cost should not exceed 20–25% of predicted LTV per segment

### LTV-Based Budget Allocation

```
Champions  ($859 avg LTV)  →  Justify up to $172–$215 CAC
Loyal      ($343 avg LTV)  →  Justify up to $69–$86 CAC
```

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.11** | Core language |
| **Pandas / NumPy** | Data manipulation |
| **Scikit-Learn** | K-Means clustering, StandardScaler |
| **Lifetimes** | BG/NBD + Gamma-Gamma modelling |
| **Matplotlib / Seaborn** | Static visualisations |
| **Plotly** | Interactive 3D scatter |
| **Streamlit** | Production web application |

---

---

| Coming Soon — Attribution Modelling | Multi-touch attribution + A/B testing |

---

*Built with Python · Streamlit · Lifetimes · Scikit-Learn*
