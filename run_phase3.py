"""
LRRS Phase 3 — Counterfactual loan-term optimisation engine.

Execution plan (per approval 2026-08-23):
  1. Load preprocessor (models/preprocessor.pkl) and model
     (models/phase2/xgboost_nolastfico_calibrated.pkl)
  2. Load test.parquet (pre-preprocessed, with engineered features)
  3. Run on 1,000-borrower random sample; save results; generate report
  4. If zero errors in sample, run on full test set; save results; generate report

Phase 3 design constraints:
  - Model: xgboost_nolastfico_calibrated.pkl (Phase 2 HIGH CONCERN verdict)
  - int_rate: held constant per borrower (Option A)
  - loan_amnt: 10 steps from $1,000 to original amount (inclusive)
  - term: both {36, 60} months evaluated per loan_amnt step
  - Feasibility: LC limits enforced

Run:
    python run_phase3.py
"""
import json
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.helpers import get_logger, ensure_dir, save_json
from src.preprocessing.target import TARGET_COL
from src.phase3.engine import score_population_batched
from src.phase3.recommend import (
    build_recommendations_from_population,
    generate_phase3_report,
)

logger = get_logger("run_phase3")

_PROCESSED_DIR = Path("data/processed")
_MODELS_DIR = Path("models")
_P3_RESULTS_DIR = Path("results/phase3")
_SAMPLE_SIZE = 1_000
_SEED = 42
_CHUNK_SIZE = 100   # borrowers per preprocessor.transform() call


def _load_artifacts():
    model_path = _MODELS_DIR / "phase2" / "xgboost_nolastfico_calibrated.pkl"
    prep_path = _MODELS_DIR / "preprocessor.pkl"
    logger.info("Loading model: %s", model_path)
    model = joblib.load(model_path)
    logger.info("Loading preprocessor: %s", prep_path)
    preprocessor = joblib.load(prep_path)
    return model, preprocessor


def _load_test_raw() -> pd.DataFrame:
    path = _PROCESSED_DIR / "test.parquet"
    logger.info("Loading test.parquet from %s", path)
    df = pd.read_parquet(path)
    # Drop target and status columns — not needed for counterfactual scoring
    for col in [TARGET_COL, "loan_status"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    logger.info("Test data: %d rows, %d columns", *df.shape)
    return df


def _save_results(results: list, label: str) -> None:
    """Persist per-borrower results as CSV (full) and JSON (first 100 for inspection)."""
    rows = []
    for r in results:
        if "error" in r:
            rows.append({
                "borrower_idx": r.get("borrower_idx"),
                "recommendation_type": "error",
                "error": r.get("error"),
            })
            continue
        orig = r.get("original", {})
        best = r.get("best_alternative") or {}
        rows.append({
            "borrower_idx": r.get("borrower_idx"),
            "orig_loan_amnt": orig.get("loan_amnt"),
            "orig_term_months": orig.get("term_months"),
            "orig_prob_default": orig.get("prob_default"),
            "best_loan_amnt": best.get("loan_amnt"),
            "best_term_months": best.get("term_months"),
            "best_installment": best.get("installment"),
            "best_prob_default": best.get("prob_default"),
            "delta_prob": r.get("delta_prob"),
            "recommendation_type": r.get("recommendation_type"),
            "n_candidates": r.get("n_candidates"),
        })

    csv_path = _P3_RESULTS_DIR / f"{label}_recommendations.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info("Per-borrower CSV saved: %s", csv_path)

    # Save first 100 results as JSON for quick inspection
    json_path = _P3_RESULTS_DIR / f"{label}_recommendations_sample.json"
    save_json(results[:100], str(json_path))
    logger.info("First 100 results (JSON): %s", json_path)


def _run_population(
    df: pd.DataFrame,
    preprocessor,
    model,
    label: str,
) -> list:
    """Score df and return per-borrower recommendation dicts."""
    t0 = time.time()
    logger.info("Scoring %d borrowers (chunk_size=%d)...", len(df), _CHUNK_SIZE)

    scores_df = score_population_batched(
        df, preprocessor, model, chunk_size=_CHUNK_SIZE,
    )

    elapsed_score = time.time() - t0
    logger.info(
        "Scoring done in %.1fs — %d scored rows for %d borrowers",
        elapsed_score, len(scores_df), len(df),
    )

    results = build_recommendations_from_population(scores_df, df)
    return results


def main():
    t_total = time.time()
    ensure_dir(_P3_RESULTS_DIR)

    logger.info("=" * 60)
    logger.info("LRRS PHASE 3 — Counterfactual Recommendation Engine")
    logger.info("=" * 60)
    logger.info("Model: xgboost_nolastfico_calibrated.pkl")
    logger.info("Reason: Phase 2 HIGH CONCERN leakage verdict (last_fico features)")
    logger.info("int_rate: held constant (Option A)")

    model, preprocessor = _load_artifacts()
    test_df = _load_test_raw()

    # ── Step 1: 1,000-borrower validation sample ──────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("STEP 1 — 1,000-borrower validation sample")
    logger.info("=" * 50)

    rng = np.random.default_rng(_SEED)
    n_sample = min(_SAMPLE_SIZE, len(test_df))
    sample_idx = sorted(rng.choice(len(test_df), size=n_sample, replace=False))
    sample_df = test_df.iloc[sample_idx].reset_index(drop=True)

    sample_results = _run_population(sample_df, preprocessor, model, label="sample")
    _save_results(sample_results, "sample")
    generate_phase3_report(
        sample_results,
        save_dir=str(_P3_RESULTS_DIR),
        population_label=f"{n_sample}-borrower validation sample",
        filename="sample_phase3_report.txt",
    )

    n_errors = sum(1 for r in sample_results if "error" in r)
    if n_errors > 0:
        logger.error(
            "Sample run had %d errors — stopping before full population run.", n_errors
        )
        logger.error("Inspect %s/sample_recommendations.csv for details.", _P3_RESULTS_DIR)
        sys.exit(1)

    logger.info("Sample run: 0 errors. Proceeding to full test population.")

    # ── Step 2: Full test population ──────────────────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("STEP 2 — Full test population (%d borrowers)", len(test_df))
    logger.info("=" * 50)

    full_results = _run_population(test_df, preprocessor, model, label="full")
    _save_results(full_results, "full")
    generate_phase3_report(
        full_results,
        save_dir=str(_P3_RESULTS_DIR),
        population_label=f"full test set ({len(test_df):,} borrowers)",
        filename="full_phase3_report.txt",
    )

    elapsed = time.time() - t_total
    logger.info("\n" + "=" * 60)
    logger.info("LRRS PHASE 3 — COMPLETE")
    logger.info("Elapsed: %.1f s (%.1f min)", elapsed, elapsed / 60)
    logger.info("=" * 60)
    logger.info("Outputs:")
    logger.info("  results/phase3/sample_recommendations.csv")
    logger.info("  results/phase3/sample_phase3_report.txt")
    logger.info("  results/phase3/full_recommendations.csv")
    logger.info("  results/phase3/full_phase3_report.txt")


if __name__ == "__main__":
    main()
