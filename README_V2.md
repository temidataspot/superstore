# Production ML Pipeline — Superstore Customer Analytics

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED?logo=docker)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?logo=github-actions)
![Lifetimes](https://img.shields.io/badge/Lifetimes-BG%2FNBD-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-K--Means-F7931E?logo=scikit-learn)
![Status](https://img.shields.io/badge/Pipeline-Passing-34d399)

> Automated MLOps pipeline that retrains BG/NBD, Gamma-Gamma and K-Means models on new transaction data — containerised with Docker and deployed via GitHub Actions CI/CD.

---

## Overview

This production pipeline extends the [Superstore Customer Analytics](https://github.com/temidataspot/superstore) project with a fully automated retraining system. Push new data → pipeline runs → models update → outputs regenerate. No manual intervention required.

```
New CSV pushed to GitHub
        ↓
GitHub Actions triggers automatically
        ↓
Train  →  Evaluate  →  Export
  ↓            ↓           ↓
.pkl files  report.json  CSVs
        ↓
Outputs committed back to repo
```

---

## Pipeline Architecture

```
superstore/
├── pipeline/
│   ├── train.py          # BG/NBD + Gamma-Gamma + K-Means training
│   ├── evaluate.py       # Model validation + sanity checks
│   └── export.py         # CSV generation for all outputs
├── scheduler.py          # Master runner — train → evaluate → export
├── Dockerfile            # Python 3.11-slim containerised environment
├── docker-compose.yml    # Pipeline + persistent monthly scheduler
├── .github/
│   └── workflows/
│       └── retrain.yml   # Auto-triggers on new data push or monthly
├── models/               # Saved .pkl model artifacts
└── outputs/              # Regenerated CSVs + model_report.json
```

---

## Models

| Model | Purpose | Library |
|---|---|---|
| **BG/NBD** | Predicts future purchase probability | `lifetimes` |
| **Gamma-Gamma** | Predicts expected monetary value | `lifetimes` |
| **K-Means** | RFM customer segmentation | `scikit-learn` |

---

## Pipeline Results

| Metric | Value |
|---|---|
| Customers modelled | 781 repeat purchasers |
| Avg predicted LTV (12M) | $723.75 |
| Median LTV | $574.64 |
| Top 10% LTV | $1,450.82 |
| RFM segments | Champions (576) · Loyal Customers (217) |
| BG/NBD penalizer | 0.01 |
| Freq–Monetary correlation | 0.0074 ✅ |
| Pipeline runtime | ~3 seconds |

---

## Outputs

Every pipeline run regenerates these files in `/outputs`:

| File | Description |
|---|---|
| `superstore_customer_master.csv` | RFM scores + segments + LTV per customer |
| `master_eda.csv` | Monthly sales by Category, Region, Segment |
| `kpi_summary.csv` | Total Revenue, Customers, Orders, AOV |
| `cohort_retention.csv` | 42-cohort retention matrix |
| `model_report.json` | Validation results + sanity checks |

---

## Running Locally

**Prerequisites:** Python 3.11+, Docker Desktop

**Run pipeline directly:**
```bash
python scheduler.py
```

**Run in Docker:**
```bash
docker build -t superstore-pipeline .
docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  superstore-pipeline
```

**Run persistent scheduler (reruns every 30 days):**
```bash
docker-compose up scheduler -d
```

---

## CI/CD — GitHub Actions

The pipeline triggers automatically on:

| Trigger | When |
|---|---|
| `push` to `Sample - Superstore.csv` | New data uploaded |
| `schedule` | 2am UTC on the 1st of every month |
| `workflow_dispatch` | Manual trigger from GitHub UI |

After each run, updated model artifacts and CSVs are committed back to the repo automatically.

To trigger manually: **GitHub repo → Actions tab → Retrain Pipeline → Run workflow**

---

## Dependencies

```
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.5.2
lifetimes==0.11.3
dill==0.3.8
```

---

## How to Update with New Data

1. Replace `Sample - Superstore.csv` with your updated file
2. Push to GitHub
3. GitHub Actions triggers automatically
4. New models + outputs generated within minutes

The pipeline requires only 4 columns: `Customer ID`, `Order Date`, `Order ID`, `Sales` — it works on any retail transaction dataset.



---

*Built by **Temi Priscilla Jokotola** · BG/NBD + Gamma-Gamma · K-Means RFM · Docker · GitHub Actions*
