"""
LRRS Research — Temporal Leakage Validation runner (Item 1, tests a–d).

Runs four statistical tests on the training set to evaluate whether
last_fico_range_low / last_fico_range_high encode post-origination credit
trajectory rather than origination-time FICO scores.

THE TEST SET IS NOT LOADED OR USED BY THIS SCRIPT.

Tests:
  (a) Outcome-stratified FICO drift        (Welch t-test, Cohen's d)
  (b) Loan-age drift correlation           (Spearman rho, permutation test)
  (c) Outcome × age interaction            (Kruskal-Wallis, trend test)
  (d) Partial correlation after OLS        (point-biserial r after residualisation)

Multiple-testing: Bonferroni correction (α = 0.05 / 4 = 0.0125)

Outputs:
  results/research/temporal_leakage/temporal_leakage_results.json
  results/research/temporal_leakage/temporal_leakage_report.txt
  results/research/temporal_leakage/plots/test_a_drift_by_outcome.png
  results/research/temporal_leakage/plots/test_b_age_vs_drift.png
  results/research/temporal_leakage/plots/test_c_drift_by_quartile.png
  results/research/temporal_leakage/plots/test_d_residual_drift.png

Run:
    python run_research_temporal_leakage.py

IMPORTANT:
  Do NOT run temporal CV (test e) without separate approval — it requires
  approximately 4 CPU-hours on this hardware.  This runner implements only
  tests (a)–(d) as approved.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.research.temporal_leakage import run_temporal_leakage_tests
from src.utils.helpers import get_logger

logger = get_logger("run_research_temporal_leakage")


def main():
    logger.info("=" * 60)
    logger.info("LRRS RESEARCH — Temporal Leakage Validation (tests a–d)")
    logger.info("Test set: NOT LOADED")
    logger.info("=" * 60)

    result = run_temporal_leakage_tests()

    verdict = result.get("overall_verdict", {})
    logger.info("\nAll tests complete.")
    logger.info(
        "Significant tests (after Bonferroni): %d / %d",
        verdict.get("n_significant_tests", "?"),
        verdict.get("n_tests_total", "?"),
    )
    logger.info("Overall evidence strength: %s", verdict.get("overall_evidence_strength", "?"))
    logger.info("\nVerdict: %s", verdict.get("verdict", ""))
    logger.info("\nOutputs:")
    logger.info("  results/research/temporal_leakage/temporal_leakage_results.json")
    logger.info("  results/research/temporal_leakage/temporal_leakage_report.txt")
    logger.info("  results/research/temporal_leakage/plots/ (4 PNG files)")
    logger.info("\nNOTE: Temporal CV (test e) NOT run — requires separate approval.")


if __name__ == "__main__":
    main()
