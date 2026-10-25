"""
LRRS Research — Phase 3 Backtest and Baseline Comparison runner (Research Item 4).

Research question:
  Does the Phase 3 counterfactual recommendation strategy provide meaningful value
  beyond a simple minimum-loan baseline?

CIRCULARITY WARNING:
  The recommendation engine is optimized against the same predictive model used to
  score candidates; therefore lower predicted risk alone is not an independent
  validation of recommendation effectiveness.

Baselines:
  BL1 = Original configuration (historical loan)
  BL2 = Minimum-loan strategy: $1,000 with best term (min risk of 36 vs 60 months)
  BL3 = Same-term reduction: $1,000 with original term held constant

What this runner does:
  - Loads the Phase 3 recommendations from results/phase3/full_recommendations.csv
  - Loads the test set from data/processed/test.parquet
  - Scores each borrower at ($1,000, 36 months) and ($1,000, 60 months)
    using the nolastfico XGBoost model (vectorised in chunks)
  - Computes per-borrower risk reduction for each strategy
  - Runs Wilcoxon signed-rank tests with rank-biserial correlation effect sizes
  - Generates plots and a written report

What this runner does NOT do:
  - Modify or re-run Phase 1–3 production artifacts
  - Retrain any model
  - Fabricate counterfactual outcomes
  - Use synthetic data for conclusions

Outputs:
  results/research/phase3_backtest/phase3_backtest_report.txt
  results/research/phase3_backtest/phase3_backtest_results.json
  results/research/phase3_backtest/phase3_backtest_per_borrower.csv
  results/research/phase3_backtest/plots/*.png

Run:
    python run_research_phase3_backtest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.research.phase3_backtest import run_phase3_backtest, CIRCULARITY_STATEMENT
from src.utils.helpers import get_logger

logger = get_logger("run_research_phase3_backtest")

_RECS_PATH = "results/phase3/full_recommendations.csv"
_TEST_PARQUET = "data/processed/test.parquet"
_PREPROCESSOR_PATH = "models/preprocessor.pkl"
_MODEL_PATH = "models/phase2/xgboost_nolastfico_calibrated.pkl"
_SAVE_DIR = "results/research/phase3_backtest"

_REQUIRED_FILES = [
    _RECS_PATH,
    _TEST_PARQUET,
    _PREPROCESSOR_PATH,
    _MODEL_PATH,
]


def _check_files() -> bool:
    ok = True
    for p in _REQUIRED_FILES:
        if not Path(p).exists():
            logger.error("Required file not found: %s", p)
            ok = False
    return ok


def main() -> None:
    logger.info("=" * 60)
    logger.info("LRRS RESEARCH — Phase 3 Backtest / Baseline Comparison")
    logger.info("Research Item 4")
    logger.info("")
    logger.info("CIRCULARITY WARNING:")
    logger.info("  %s", CIRCULARITY_STATEMENT)
    logger.info("")
    logger.info("Test set: accessed READ-ONLY for scoring and observed outcomes.")
    logger.info("Phase 3 artifacts: accessed READ-ONLY; not modified.")
    logger.info("=" * 60)

    if not _check_files():
        logger.error("One or more required files missing — aborting.")
        sys.exit(1)

    results = run_phase3_backtest(
        recs_path=_RECS_PATH,
        test_parquet=_TEST_PARQUET,
        preprocessor_path=_PREPROCESSOR_PATH,
        model_path=_MODEL_PATH,
        save_dir=_SAVE_DIR,
        chunk_size=5000,
    )

    m = results["metrics"]
    rs = m["risk_stats"]
    st = m["structural_analysis"]
    la = m["loan_amount_stats"]
    tests = m["statistical_tests"]

    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info("Borrowers:                  %d", m["n_borrowers"])
    logger.info("")
    logger.info("Mean predicted default probability:")
    logger.info("  BL1 (original):           %.4f", rs["BL1_original"]["mean"])
    logger.info("  Phase 3:                  %.4f", rs["phase3"]["mean"])
    logger.info("  BL2 (min-loan best-term): %.4f", rs["BL2_minloan_bestterm"]["mean"])
    logger.info("  BL3 (min-loan same-term): %.4f", rs["BL3_minloan_sameterm"]["mean"])
    logger.info("")
    logger.info("Structural analysis:")
    logger.info("  Phase 3 at $1,000 min:    %d  (%.1f%%)",
                st["n_at_min_loan_1000"], st["pct_at_min_loan"])
    logger.info("  Phase 3 changes term:     %d  (%.1f%%)",
                st["n_phase3_changes_term"], st["pct_phase3_changes_term"])
    logger.info("  Phase 3 differs from BL2: %d  (%.1f%%)",
                st["n_phase3_differs_from_BL2"], st["pct_phase3_differs_from_BL2"])
    logger.info("")
    logger.info("Loan amount retained (Phase 3):")
    logger.info("  Mean original:            $%.2f", la["mean_original"])
    logger.info("  Mean recommended:         $%.2f", la["mean_phase3"])
    logger.info("  Mean retained (%%):         %.2f%%", la["mean_retained_pct"])
    logger.info("")
    logger.info("Statistical tests (Phase 3 < baseline):")
    for k, label in [
        ("phase3_vs_BL2_wilcoxon", "Phase 3 vs BL2"),
        ("phase3_vs_BL3_wilcoxon", "Phase 3 vs BL3"),
    ]:
        t = tests.get(k, {})
        p = t.get("p_value", "n/a")
        r = t.get("rank_biserial_r", "n/a")
        n_nz = t.get("n_nonzero", "n/a")
        p_fmt = f"{p:.4e}" if isinstance(p, float) else str(p)
        r_fmt = f"{r:.4f}" if isinstance(r, float) else str(r)
        logger.info("  %s: p=%s  r_rb=%s  n_nonzero=%s",
                    label, p_fmt, r_fmt, n_nz)
    logger.info("")
    logger.info("Outputs:")
    for k, v in results["paths"].items():
        logger.info("  %s: %s", k, v)
    logger.info("")
    logger.info("NOTE: p-values reflect mathematical construction (circularity),")
    logger.info("      not independent causal evidence. See report for full discussion.")


if __name__ == "__main__":
    main()
