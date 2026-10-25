"""
LRRS Research — Feature Timestamp Audit runner (Item 2).

Classifies Phase 1 model features by measurement timing relative to loan
origination, using structural heuristics and Phase 2 empirical evidence.

Outputs:
  results/research/feature_timestamp_audit.json
  results/research/feature_timestamp_report.txt

IMPORTANT:
  The LendingClub data dictionary is NOT included in this repository.
  If it is not present, this script will identify what is needed and
  complete the analysis using available evidence only.  It will NOT
  fabricate citations or download anything.

Run:
    python run_research_feature_audit.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.research.feature_timestamp_audit import run_feature_timestamp_audit
from src.utils.helpers import get_logger

logger = get_logger("run_research_feature_audit")


def main():
    logger.info("=" * 60)
    logger.info("LRRS RESEARCH — Feature Timestamp Audit")
    logger.info("=" * 60)

    result = run_feature_timestamp_audit(
        feature_audit_path="results/reports/feature_audit.json",
        data_dict_path="LCDataDictionary.xlsx",
        save_dir="results/research",
    )

    summary = result.get("summary", {})
    meta = result.get("metadata", {})
    counts = summary.get("timing_value_counts", {})

    logger.info("\nAudit complete.")
    logger.info("Data dictionary available: %s", meta.get("data_dictionary_available"))
    logger.info("Features classified: %d", meta.get("n_included_features", 0))
    logger.info("Timing distribution:")
    for timing_val, count in sorted(counts.items(), key=lambda x: -x[1]):
        if count > 0:
            logger.info("  %-35s %d", timing_val, count)

    logger.info("\nOutputs:")
    logger.info("  results/research/feature_timestamp_audit.json")
    logger.info("  results/research/feature_timestamp_report.txt")


if __name__ == "__main__":
    main()
