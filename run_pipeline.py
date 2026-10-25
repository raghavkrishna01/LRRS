"""
LRRS Phase 1 — Main Pipeline Runner

Executes all steps in order:
  1.  Load raw data
  2.  Profile dataset
  3.  Create binary target + drop ambiguous rows
  4.  Feature audit (leakage detection)
  5.  Engineer features
  6.  Train / validation / test split (temporal)
  7.  Fit preprocessing pipeline (training data only)
  8.  Train Random Forest
  9.  Train XGBoost
  10. Evaluate both models (validation + test)
  11. Calibrate probabilities (validation set only)
  12. Cost-sensitive threshold selection (validation set only)
  13. Final test-set evaluation at optimal threshold
  14. Feature importance
  15. Phase 3 compatibility analysis
  16. Generate final report

Run:
    python run_pipeline.py
    python run_pipeline.py --max-rows 100000   # quick test run
    python run_pipeline.py --skip-tuning        # use default hyperparams
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.helpers import load_config, get_logger, save_json
from src.data.loader import load_raw_data
from src.data.profiler import profile_dataset
from src.preprocessing.target import create_binary_target, TARGET_COL
from src.preprocessing.feature_audit import (
    save_feature_audit, get_usable_features, get_excluded_features, get_actionable_features
)
from src.features.engineering import engineer_features, print_feature_catalogue
from src.splitting.splitter import temporal_split, save_splits
from src.preprocessing.pipeline import LendingClubPreprocessor
from src.models.train_rf import train_random_forest, load_random_forest
from src.models.train_xgb import train_xgboost, load_xgboost
from src.evaluation.metrics import (
    evaluate_model, select_cost_sensitive_threshold,
    compare_models, evaluate_feature_importance,
)
from src.calibration.calibrate import calibrate_model, compare_calibration
from src.evaluation.phase3_analysis import run_phase3_analysis
from src.evaluation.final_report import generate_final_report

logger = get_logger("run_pipeline")


def parse_args():
    parser = argparse.ArgumentParser(description="LRRS Phase 1 Pipeline")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path to config file")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limit number of rows loaded (for testing)")
    parser.add_argument("--skip-tuning", action="store_true",
                        help="Skip hyperparameter tuning and use default params")
    parser.add_argument("--skip-profiling", action="store_true",
                        help="Skip data profiling (saves time on re-runs)")
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    logger.info("=" * 60)
    logger.info("LRRS PHASE 1 PIPELINE — START")
    logger.info("=" * 60)

    config = load_config(args.config)
    seed = config["project"]["random_seed"]
    np.random.seed(seed)

    # ── STEP 1: Load data ──────────────────────────────────────────────
    logger.info("\n─── STEP 1: Loading raw data ───")
    df_raw = load_raw_data(config=config, max_rows=args.max_rows)
    logger.info("Raw data loaded: %d rows × %d cols", *df_raw.shape)

    # ── STEP 2: Profile ────────────────────────────────────────────────
    if not args.skip_profiling:
        logger.info("\n─── STEP 2: Profiling dataset ───")
        profile = profile_dataset(df_raw, save_dir="results/reports")
    else:
        logger.info("Step 2: Profiling skipped")
        profile = {}

    # ── STEP 3: Target creation ────────────────────────────────────────
    logger.info("\n─── STEP 3: Creating binary target ───")
    df = create_binary_target(df_raw)
    logger.info(
        "After target creation: %d rows  Default rate: %.2f%%",
        len(df), df[TARGET_COL].mean() * 100,
    )

    # ── STEP 4: Feature audit ──────────────────────────────────────────
    logger.info("\n─── STEP 4: Feature audit (leakage detection) ───")
    audit = save_feature_audit(df.columns.tolist(), save_dir="results/reports")
    usable_cols = get_usable_features(df.columns.tolist())
    excluded = get_excluded_features(df.columns.tolist())
    actionable = get_actionable_features(df.columns.tolist())
    logger.info(
        "Usable features: %d  Excluded: %d  Actionable: %d",
        len(usable_cols), len(excluded), len(actionable),
    )

    # Keep only: usable features + target + issue_d for splitting
    keep_cols = list(set(usable_cols + [TARGET_COL, "issue_d", "loan_status"]))
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # ── STEP 5: Feature engineering ────────────────────────────────────
    logger.info("\n─── STEP 5: Feature engineering ───")
    print_feature_catalogue()
    df = engineer_features(df)
    logger.info("After engineering: %d rows × %d cols", *df.shape)

    # ── STEP 6: Train/val/test split ───────────────────────────────────
    logger.info("\n─── STEP 6: Train / Validation / Test split ───")
    logger.info("Method: Temporal (chronological by issue_d)")
    logger.info("Rationale: Simulates real-world deployment on future loans")

    train_df, val_df, test_df = temporal_split(
        df,
        date_col="issue_d",
        train_ratio=config["splitting"]["train_ratio"],
        val_ratio=config["splitting"]["val_ratio"],
        target_col=TARGET_COL,
    )
    save_splits(train_df, val_df, test_df, save_dir="data/processed")

    # ── STEP 7: Preprocessing ──────────────────────────────────────────
    logger.info("\n─── STEP 7: Fitting preprocessing pipeline (training data only) ───")

    # Determine final feature columns (exclude target, loan_status, issue_d)
    meta_cols = {TARGET_COL, "loan_status", "issue_d"}
    feature_cols = [
        c for c in train_df.columns
        if c not in meta_cols and not pd.api.types.is_datetime64_any_dtype(train_df[c])
    ]

    preprocessor = LendingClubPreprocessor(config=config)

    # Fit and transform training data
    X_train = preprocessor.fit_transform(
        train_df, feature_cols=feature_cols, issue_date_col="issue_d"
    )
    y_train = train_df[TARGET_COL].reset_index(drop=True)

    # Transform validation and test (using fitted parameters — NO leakage)
    X_val = preprocessor.transform(val_df, issue_date_col="issue_d")
    y_val = val_df[TARGET_COL].reset_index(drop=True)

    X_test = preprocessor.transform(test_df, issue_date_col="issue_d")
    y_test = test_df[TARGET_COL].reset_index(drop=True)

    feature_names = preprocessor.feature_names
    logger.info(
        "Preprocessed shapes — train: %s  val: %s  test: %s",
        X_train.shape, X_val.shape, X_test.shape,
    )

    # Save preprocessor
    preprocessor.save("models/preprocessor.pkl")
    save_json({"feature_names": feature_names}, "models/feature_names.json")

    # Save processed arrays
    pd.concat([X_train, y_train.rename(TARGET_COL)], axis=1).to_parquet(
        "data/processed/train_preprocessed.parquet", index=False
    )
    pd.concat([X_val, y_val.rename(TARGET_COL)], axis=1).to_parquet(
        "data/processed/val_preprocessed.parquet", index=False
    )
    pd.concat([X_test, y_test.rename(TARGET_COL)], axis=1).to_parquet(
        "data/processed/test_preprocessed.parquet", index=False
    )

    # ── STEP 8: Train Random Forest ────────────────────────────────────
    logger.info("\n─── STEP 8: Training Random Forest ───")
    if args.skip_tuning:
        logger.info("Skipping tuning — using default hyperparams")
        from sklearn.ensemble import RandomForestClassifier
        rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)
        rf_params = rf_model.get_params()
        import joblib
        joblib.dump(rf_model, "models/random_forest.pkl")
        save_json({"best_params": rf_params, "best_cv_roc_auc": None},
                  "models/random_forest_config.json")
    else:
        rf_model, rf_params = train_random_forest(X_train, y_train, config)

    # ── STEP 9: Train XGBoost ──────────────────────────────────────────
    logger.info("\n─── STEP 9: Training XGBoost ───")
    if args.skip_tuning:
        import xgboost as xgb
        from src.models.train_xgb import compute_scale_pos_weight
        spw = compute_scale_pos_weight(y_train)
        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, objective="binary:logistic",
            eval_metric="aucpr", tree_method="hist",
            random_state=seed, verbosity=0, n_jobs=-1,
        )
        xgb_model.fit(X_train, y_train)
        xgb_model.save_model("models/xgboost.json")
        xgb_params = xgb_model.get_params()
        save_json({"best_params": xgb_params, "scale_pos_weight": spw},
                  "models/xgboost_config.json")
    else:
        xgb_model, xgb_params = train_xgboost(X_train, y_train, config)

    # ── STEP 10: Evaluate (validation set, default threshold) ──────────
    logger.info("\n─── STEP 10: Evaluating on VALIDATION set (threshold=0.50) ───")
    rf_val_prob = rf_model.predict_proba(X_val)[:, 1]
    xgb_val_prob = xgb_model.predict_proba(X_val)[:, 1]

    rf_val_metrics = evaluate_model(
        "random_forest_val", y_val, rf_val_prob, threshold=0.50
    )
    xgb_val_metrics = evaluate_model(
        "xgboost_val", y_val, xgb_val_prob, threshold=0.50
    )

    # ── STEP 11: Calibrate (validation set) ────────────────────────────
    logger.info("\n─── STEP 11: Calibrating probabilities (validation set) ───")
    logger.info("Method: Platt scaling (sigmoid) fitted on validation set")
    logger.info("Rationale: Produces smooth, meaningful probability estimates")

    rf_cal_model, rf_cal_metrics = calibrate_model(
        rf_model, X_val, y_val, X_train, y_train,
        model_name="random_forest",
    )
    xgb_cal_model, xgb_cal_metrics = calibrate_model(
        xgb_model, X_val, y_val, X_train, y_train,
        model_name="xgboost",
    )
    compare_calibration(rf_cal_metrics, xgb_cal_metrics)

    # Calibrated validation probabilities
    rf_val_cal_prob = rf_cal_model.predict_proba(X_val)[:, 1]
    xgb_val_cal_prob = xgb_cal_model.predict_proba(X_val)[:, 1]

    # ── STEP 12: Cost-sensitive threshold (validation set only) ────────
    logger.info("\n─── STEP 12: Cost-sensitive threshold selection (validation set) ───")
    fn_fp_ratio = config["evaluation"]["fn_fp_cost_ratio"]
    logger.info(
        "FN/FP cost ratio: %.1f  — Missing a defaulter is %.1fx more costly",
        fn_fp_ratio, fn_fp_ratio,
    )
    logger.info(
        "Rationale: A missed default may mean full principal loss; "
        "a false positive only costs lost interest margin on one loan."
    )

    rf_threshold = select_cost_sensitive_threshold(
        y_val, rf_val_cal_prob, fn_fp_cost_ratio=fn_fp_ratio, model_name="random_forest"
    )
    xgb_threshold = select_cost_sensitive_threshold(
        y_val, xgb_val_cal_prob, fn_fp_cost_ratio=fn_fp_ratio, model_name="xgboost"
    )

    # ── STEP 13: Final test-set evaluation ─────────────────────────────
    logger.info("\n─── STEP 13: FINAL evaluation on TEST SET ───")
    logger.info("NOTE: Test set has not been seen during any prior step.")

    rf_test_prob = rf_cal_model.predict_proba(X_test)[:, 1]
    xgb_test_prob = xgb_cal_model.predict_proba(X_test)[:, 1]

    # At default threshold
    rf_test_metrics_50 = evaluate_model(
        "random_forest", y_test, rf_test_prob, threshold=0.50
    )
    xgb_test_metrics_50 = evaluate_model(
        "xgboost", y_test, xgb_test_prob, threshold=0.50
    )

    # At cost-sensitive threshold
    logger.info("\nTest evaluation at cost-sensitive thresholds:")
    rf_test_metrics_cs = evaluate_model(
        "random_forest_cost_sensitive", y_test, rf_test_prob, threshold=rf_threshold
    )
    xgb_test_metrics_cs = evaluate_model(
        "xgboost_cost_sensitive", y_test, xgb_test_prob, threshold=xgb_threshold
    )

    compare_models(
        {
            "Random Forest (0.50)": rf_test_metrics_50,
            "XGBoost (0.50)": xgb_test_metrics_50,
        }
    )
    compare_models(
        {
            f"RF (thresh={rf_threshold:.3f})": rf_test_metrics_cs,
            f"XGB (thresh={xgb_threshold:.3f})": xgb_test_metrics_cs,
        }
    )

    # ── STEP 14: Feature importance ────────────────────────────────────
    logger.info("\n─── STEP 14: Feature importance ───")
    rf_fi = evaluate_feature_importance(rf_model, feature_names, "random_forest")
    xgb_fi = evaluate_feature_importance(xgb_model, feature_names, "xgboost")

    if not rf_fi.empty:
        logger.info("Top 10 RF features:\n%s", rf_fi.head(10).to_string(index=False))
    if not xgb_fi.empty:
        logger.info("Top 10 XGB features:\n%s", xgb_fi.head(10).to_string(index=False))

    # ── STEP 15: Phase 3 compatibility analysis ─────────────────────────
    logger.info("\n─── STEP 15: Phase 3 compatibility analysis ───")
    run_phase3_analysis(save_dir="results")

    # ── STEP 16: Final report ──────────────────────────────────────────
    logger.info("\n─── STEP 16: Generating Phase 1 final report ───")
    generate_final_report(
        metrics_dir="results/metrics",
        save_dir="results/reports",
    )

    elapsed = time.time() - t0
    logger.info("\n" + "=" * 60)
    logger.info("LRRS PHASE 1 PIPELINE — COMPLETE")
    logger.info("Elapsed time: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)
    logger.info("=" * 60)
    logger.info("\nSaved outputs:")
    logger.info("  models/preprocessor.pkl")
    logger.info("  models/random_forest.pkl")
    logger.info("  models/xgboost.json")
    logger.info("  models/random_forest_calibrated.pkl")
    logger.info("  models/xgboost_calibrated.pkl")
    logger.info("  data/processed/  (train/val/test splits)")
    logger.info("  results/metrics/ (all metric JSONs)")
    logger.info("  results/plots/   (all plots)")
    logger.info("  results/reports/ (all text reports)")


if __name__ == "__main__":
    main()
