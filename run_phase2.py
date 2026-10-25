"""
LRRS Phase 2 — Main runner.

Executes all six Phase 2 deliverables in order:
  D2 (calibration quality)    — no retraining
  D3 (subgroup analysis)      — no retraining
  D5 (temporal cohort)        — no retraining
  D4 (grade robustness)       — one XGBoost variant fit + permutation importance
  D1 (leakage investigation)  — one XGBoost variant fit
  D6 (Phase 2 summary report) — no computation

Reuses all Phase 1 artifacts. Does not modify Phase 1 results.
Phase 2 outputs saved under results/phase2/ and models/phase2/.

Run:
    python run_phase2.py
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.helpers import load_config, get_logger, ensure_dir
from src.preprocessing.target import TARGET_COL

from src.phase2.calibration_quality import run_calibration_quality
from src.phase2.subgroup import run_subgroup_analysis
from src.phase2.temporal_cohort import run_temporal_cohort
from src.phase2.grade_robustness import run_grade_robustness
from src.phase2.leakage import run_leakage_investigation
from src.phase2.report import generate_phase2_report

logger = get_logger("run_phase2")

_PROCESSED_DIR = Path("data/processed")
_MODELS_DIR = Path("models")
_P2_MODELS_DIR = Path("models/phase2")
_P2_RESULTS_DIR = Path("results/phase2")


def _load_split(name: str):
    path = _PROCESSED_DIR / f"{name}_preprocessed.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Preprocessed split not found: {path}")
    df = pd.read_parquet(path)
    y = df[TARGET_COL].reset_index(drop=True)
    X = df.drop(columns=[TARGET_COL]).reset_index(drop=True)
    return X, y


def main():
    t0 = time.time()

    config = load_config("configs/config.yaml")
    seed = config["project"]["random_seed"]
    np.random.seed(seed)

    logger.info("=" * 60)
    logger.info("LRRS PHASE 2 — All Deliverables")
    logger.info("=" * 60)

    ensure_dir(_P2_MODELS_DIR)
    ensure_dir(_P2_RESULTS_DIR / "metrics")
    ensure_dir(_P2_RESULTS_DIR / "plots")
    ensure_dir(_P2_RESULTS_DIR / "reports")

    # ── Load Phase 1 artifacts ─────────────────────────────────────────────
    logger.info("Loading Phase 1 artifacts (read-only)")

    X_train, y_train = _load_split("train")
    X_val,   y_val   = _load_split("val")
    X_test,  y_test  = _load_split("test")

    with open(_MODELS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)["feature_names"]

    with open(_MODELS_DIR / "xgboost_config.json") as f:
        xgb_cfg = json.load(f)

    best_params = xgb_cfg["best_params"]
    scale_pos_weight = xgb_cfg["scale_pos_weight"]

    logger.info(
        "Splits: train=%s  val=%s  test=%s", X_train.shape, X_val.shape, X_test.shape
    )
    logger.info("Phase 1 XGB best params: %s", best_params)
    logger.info("scale_pos_weight: %.4f", scale_pos_weight)

    # Load Phase 1 calibrated models
    xgb_cal_model = joblib.load(_MODELS_DIR / "xgboost_calibrated.pkl")
    rf_cal_model  = joblib.load(_MODELS_DIR / "random_forest_calibrated.pkl")

    # Load base (uncalibrated) XGBoost for permutation importance
    xgb_base_model = xgb.XGBClassifier()
    xgb_base_model.load_model(str(_MODELS_DIR / "xgboost.json"))

    logger.info("Phase 1 models loaded.")

    # ── D2: Test-set calibration quality (no retraining) ──────────────────
    logger.info("\n" + "=" * 50)
    logger.info("DELIVERABLE 2 — Test-set calibration quality")
    logger.info("=" * 50)
    run_calibration_quality(
        xgb_cal_model, rf_cal_model, X_test, y_test,
        save_dir=str(_P2_RESULTS_DIR),
    )

    # ── D3: Subgroup analysis (no retraining) ─────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("DELIVERABLE 3 — Joint vs individual subgroup")
    logger.info("=" * 50)
    # Phase 1 cost-sensitive threshold from threshold_analysis.json
    xgb_thresh_path = _MODELS_DIR / ".." / "results" / "metrics" / "xgboost_threshold_analysis.json"
    optimal_threshold = 0.1725  # Phase 1 default; overridden below if file exists
    try:
        with open(Path("results/metrics/xgboost_threshold_analysis.json")) as f:
            optimal_threshold = json.load(f)["optimal_threshold"]
    except Exception:
        logger.warning("Could not load optimal threshold; using %.4f", optimal_threshold)

    run_subgroup_analysis(
        xgb_cal_model, X_test, y_test,
        optimal_threshold=optimal_threshold,
        save_dir=str(_P2_RESULTS_DIR),
    )

    # ── D5: Temporal cohort (no retraining) ───────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("DELIVERABLE 5 — Temporal cohort performance")
    logger.info("=" * 50)
    run_temporal_cohort(
        xgb_cal_model, X_test, y_test,
        test_raw_path=str(_PROCESSED_DIR / "test.parquet"),
        save_dir=str(_P2_RESULTS_DIR),
    )

    # ── D4: Grade robustness (permutation importance + one variant fit) ────
    logger.info("\n" + "=" * 50)
    logger.info("DELIVERABLE 4 — Grade robustness")
    logger.info("=" * 50)
    run_grade_robustness(
        xgb_base_model, xgb_cal_model,
        X_train, y_train, X_val, y_val, X_test, y_test,
        feature_names=feature_names,
        best_params=best_params,
        scale_pos_weight=scale_pos_weight,
        save_dir=str(_P2_RESULTS_DIR),
        models_dir=str(_P2_MODELS_DIR),
    )

    # ── D1: Leakage investigation (one variant fit) ────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("DELIVERABLE 1 — Last-FICO leakage investigation")
    logger.info("=" * 50)
    run_leakage_investigation(
        xgb_base_model, xgb_cal_model,
        X_train, y_train, X_val, y_val, X_test, y_test,
        feature_names=feature_names,
        best_params=best_params,
        scale_pos_weight=scale_pos_weight,
        save_dir=str(_P2_RESULTS_DIR),
        models_dir=str(_P2_MODELS_DIR),
    )

    # ── D6: Phase 2 summary report ─────────────────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("DELIVERABLE 6 — Phase 2 final report")
    logger.info("=" * 50)
    generate_phase2_report(
        phase2_metrics_dir=str(_P2_RESULTS_DIR / "metrics"),
        phase1_metrics_dir="results/metrics",
        save_dir=str(_P2_RESULTS_DIR / "reports"),
    )

    elapsed = time.time() - t0
    logger.info("\n" + "=" * 60)
    logger.info("LRRS PHASE 2 — COMPLETE")
    logger.info("Elapsed: %.1f s (%.1f min)", elapsed, elapsed / 60)
    logger.info("=" * 60)
    logger.info("\nPhase 2 artifacts:")
    logger.info("  models/phase2/                (variant XGBoost models)")
    logger.info("  results/phase2/metrics/       (all Phase 2 metric JSONs)")
    logger.info("  results/phase2/plots/         (calibration, subgroup, temporal, grade plots)")
    logger.info("  results/phase2/reports/       (all Phase 2 reports + summary)")


if __name__ == "__main__":
    main()
