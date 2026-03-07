"""
scheduler.py

Master pipeline runner.
Runs train → evaluate → export in sequence.
Can be triggered manually, via Docker, or GitHub Actions.
"""

import sys
import os
import logging
from datetime import datetime

# Logging setup 
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)s  %(message)s",
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", mode="a")
    ]
)
log = logging.getLogger(__name__)

# Add repo root to path 
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from pipeline import train, evaluate, export

def run_pipeline():
    start = datetime.utcnow()
    log.info("=" * 55)
    log.info("  SUPERSTORE PIPELINE STARTING")
    log.info(f"  {start.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    log.info("=" * 55)

    try:
        # Step 1: Train 
        log.info("\n STEP 1/3 — TRAINING MODELS")
        df, rfm, ltv_sum, bgf, ggf, kmeans, scaler = train.run()
        log.info("   Step 1 complete ")

        # Step 2: Evaluate 
        log.info("\n STEP 2/3 — EVALUATING MODELS")
        meta   = {
            "trained_at":    datetime.utcnow().isoformat(),
            "best_k":        kmeans.get_params()["n_clusters"],
            "customers_ltv": int(len(ltv_sum)),
            "avg_ltv":       float(round(ltv_sum["LTV_12M"].mean(), 2))    if "LTV_12M" in ltv_sum.columns else None,
            "median_ltv":    float(round(ltv_sum["LTV_12M"].median(), 2))  if "LTV_12M" in ltv_sum.columns else None,
            "top10_ltv":     float(round(ltv_sum["LTV_12M"].quantile(0.9), 2)) if "LTV_12M" in ltv_sum.columns else None,
        }
        report = evaluate.run(meta=meta)
        log.info("   Step 2 complete ")

        # Step 3: Export 
        log.info("\n STEP 3/3 — EXPORTING OUTPUTS")
        models = {
            "kmeans_model.pkl": kmeans,
            "rfm_scaler.pkl":   scaler,
            "bgf_model.pkl":    bgf,
            "ggf_model.pkl":    ggf,
        }
        export.run(models=models)
        log.info("   Step 3 complete ")

        # Summary 
        elapsed = (datetime.utcnow() - start).seconds
        log.info("\n" + "=" * 55)
        log.info("  PIPELINE COMPLETE ")
        log.info(f"  Runtime: {elapsed}s")
        log.info(f"  Checks passed: {report['all_checks_passed']}")
        if meta.get("avg_ltv"):
            log.info(f"  Avg LTV:  ${meta['avg_ltv']:,.2f}")
        if meta.get("customers_ltv"):
            log.info(f"  Customers modelled: {meta['customers_ltv']:,}")
        log.info("=" * 55)
        return True

    except Exception as e:
        log.error(f"\n PIPELINE FAILED: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
