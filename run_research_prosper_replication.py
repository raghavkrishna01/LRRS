"""
LRRS Research — Prosper Methodological Replication runner (Research Item 3).

Replication type: METHODOLOGICAL REPLICATION
Dataset: Prosper Marketplace Loan Data (Udacity distribution, 113,937 loans)

What this runner does:
  - Loads Prosper dataset from data/raw/prosperLoanData.csv
  - Filters to post-2009 resolved loans (consistent ProsperRating regime)
  - Applies temporal 70/15/15 split on LoanOriginationDate
  - Trains XGBoost using only application-time features (n=49)
  - Fits Platt calibration on validation set
  - Evaluates on test set (first and only access)
  - Saves report, metrics, and model to results/research/second_dataset/

What this runner does NOT do:
  - Modify any Phase 1–3 artifacts (those remain read-only)
  - Test the last_fico leakage (Prosper has no equivalent post-origination FICO)
  - Use the test set for any hyperparameter selection

Run:
    python run_research_prosper_replication.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.research.prosper_replication import (
    run_prosper_replication,
    REPLICATION_TYPE,
    LC_NOLASTFICO_BASELINE,
)
from src.utils.helpers import get_logger

logger = get_logger("run_research_prosper_replication")

_CSV_PATH = "data/raw/prosperLoanData.csv"
_SAVE_DIR = "results/research/second_dataset"


def main() -> None:
    logger.info("=" * 60)
    logger.info("LRRS RESEARCH — Prosper Methodological Replication")
    logger.info("Replication type: %s", REPLICATION_TYPE)
    logger.info("Source: Prosper Marketplace Loan Data (Udacity distribution)")
    logger.info("LC nolastfico baseline: ROC-AUC=%.4f  PR-AUC=%.4f",
                LC_NOLASTFICO_BASELINE["roc_auc"], LC_NOLASTFICO_BASELINE["pr_auc"])
    logger.info("Test set: NOT LOADED — locked until final evaluation")
    logger.info("=" * 60)

    if not Path(_CSV_PATH).exists():
        logger.error("Prosper CSV not found at %s", _CSV_PATH)
        logger.error("Download it from: https://s3.amazonaws.com/udacity-hosted-downloads/ud651/prosperLoanData.csv")
        sys.exit(1)

    results = run_prosper_replication(
        csv_path=_CSV_PATH,
        save_dir=_SAVE_DIR,
        seed=42,
        n_iter=15,
        cv_folds=5,
    )

    lc = LC_NOLASTFICO_BASELINE
    test_cal = results["test_metrics_calibrated"]
    ds = results["data_summary"]
    ss = results["split_summary"]

    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info("Dataset: %d total resolved post-2009 loans", ds["n_total"])
    logger.info("  Train: %d  Val: %d  Test: %d",
                ss["n_train"], ss["n_val"], ss["n_test"])
    logger.info("  Default rate (overall): %.4f", ds["default_rate"])
    logger.info("")
    logger.info("TEST SET METRICS (calibrated XGBoost):")
    logger.info("  ROC-AUC: %.4f  (LC nolastfico: %.4f  diff: %+.4f)",
                test_cal["roc_auc"], lc["roc_auc"],
                test_cal["roc_auc"] - lc["roc_auc"])
    logger.info("  PR-AUC:  %.4f  (LC nolastfico: %.4f  diff: %+.4f)",
                test_cal["pr_auc"], lc["pr_auc"],
                test_cal["pr_auc"] - lc["pr_auc"])
    logger.info("  Brier:   %.4f", test_cal["brier_score"])
    logger.info("")
    logger.info("Outputs written to %s/", _SAVE_DIR)
    logger.info("  prosper_replication_report.txt")
    logger.info("  prosper_replication_results.json")
    logger.info("  prosper_feature_audit.json")
    logger.info("  prosper_xgb_model.json")
    logger.info("  prosper_xgb_calibrated.pkl")


if __name__ == "__main__":
    main()
