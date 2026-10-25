"""
LRRS Research — SHAP Explainability runner (Research Item 5).

Non-causality statement:
  SHAP values explain model-predicted risk contributions.
  They do NOT establish causal effects on real-world default probability.

What this runner does:
  A. Global SHAP: mean |SHAP| importance on 2,000-borrower stratified sample.
  B. Local SHAP: waterfall explanations for 5 representative borrowers.
  C. Counterfactual SHAP: delta_SHAP between original and Phase 3 recommendation
     for up to 10 representative borrowers.
  - Compares nolastfico model importance with Phase 1 model (context only).
  - Generates plots and written report.

What this runner does NOT do:
  - Retrain any model.
  - Modify Phase 1–3 models or artifacts.
  - Include last_fico features in nolastfico explanations.
  - Fabricate explanations.
  - Claim causal effects.

Outputs:
  results/research/shap/shap_report.txt
  results/research/shap/shap_global_importance.json
  results/research/shap/shap_local_explanations.json
  results/research/shap/shap_counterfactual_explanations.json
  results/research/shap/plots/

Run:
    python run_research_shap_explainability.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.research.shap_explainability import (
    run_shap_explainability,
    SHAP_NONCAUSAL_DISCLAIMER,
    PRIMARY_MODEL_PATH,
    GLOBAL_SAMPLE_N,
    GLOBAL_SAMPLE_SEED,
    LAST_FICO_COLS,
)
from src.utils.helpers import get_logger

logger = get_logger("run_research_shap_explainability")

_REQUIRED_FILES = [
    "data/processed/test.parquet",
    "models/preprocessor.pkl",
    "models/phase2/xgboost_nolastfico_calibrated.pkl",
    "results/phase3/full_recommendations.csv",
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
    logger.info("LRRS RESEARCH — SHAP Explainability (Research Item 5)")
    logger.info("")
    logger.info("NON-CAUSALITY STATEMENT:")
    logger.info("  %s", SHAP_NONCAUSAL_DISCLAIMER)
    logger.info("")
    logger.info("Primary model: %s", PRIMARY_MODEL_PATH)
    logger.info("last_fico features excluded: %s", sorted(LAST_FICO_COLS))
    logger.info("Global sample: n=%d  seed=%d", GLOBAL_SAMPLE_N, GLOBAL_SAMPLE_SEED)
    logger.info("=" * 60)

    if not _check_files():
        logger.error("Required files missing — aborting.")
        sys.exit(1)

    results = run_shap_explainability(
        save_dir="results/research/shap",
        global_sample_n=GLOBAL_SAMPLE_N,
        seed=GLOBAL_SAMPLE_SEED,
        n_cf_cases=10,
    )

    m = results["metadata"]
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info("Model features: %d (last_fico excluded: %s)",
                m["n_features"], m["last_fico_excluded"])
    logger.info("Global sample: %d  seed=%d", m["global_sample_n"], m["seed"])
    logger.info("Runtime: %.1f seconds", results["runtime_secs"])
    logger.info("")
    logger.info("Top 10 features by mean |SHAP|:")
    for row in results["global_top15"][:10]:
        logger.info("  %2d. %-40s %.5f", row["rank"], row["feature"], row["mean_abs_shap"])
    logger.info("")
    logger.info("Local explanations: %d borrowers", len(results["local_explanations"]))
    for local in results["local_explanations"]:
        logger.info("  [%s] P(default)=%.4f  row=%d",
                    local["borrower_label"], local["calibrated_prob"], local["row_idx"])
    logger.info("")
    logger.info("Counterfactual explanations: %d cases", len(results["counterfactual_explanations"]))
    for cf in results["counterfactual_explanations"]:
        o = cf["original_config"]
        r = cf["recommended_config"]
        summary = (
            f"  [{cf['borrower_label']}] "
            f"${o['loan_amnt']:,.0f}/{o['term_months']}mo -> "
            f"${r['loan_amnt']:,.0f}/{r['term_months']}mo  "
            f"dP={cf['prob_delta']:+.4f}"
        )
        logger.info(summary)
    logger.info("")
    logger.info("Outputs:")
    for k, v in results["paths"].items():
        logger.info("  %s: %s", k, v)
    logger.info("")
    logger.info("NOTE: SHAP explains model behavior only. Not causal.")


if __name__ == "__main__":
    main()
