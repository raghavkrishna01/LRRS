"""
LRRS Research — Authoritative Feature Timestamp Audit runner (Research Item 2).

Classifies key features by measurement timing using the official LendingClub
data dictionary (LCDataDictionary.csv) and corroborating web-accessible sources.

This runner upgrades prior heuristic classifications to 'Confirmed' where
authoritative documentation supports it.

Key finding:
  last_fico_range_low/high — CONFIRMED POST-ORIGINATION
    (data dict: 'last FICO pulled'; corroborated by last_credit_pull_d field
     and web-accessible secondary sources)

  fico_range_low/high — CONFIRMED ORIGINATION-TIME
    (data dict: 'FICO at loan origination')

Outputs:
  results/research/timestamp_audit/timestamp_audit.json
  results/research/timestamp_audit/timestamp_audit_report.txt
  results/research/timestamp_audit/timestamp_audit_sources.txt

Run:
    python run_research_timestamp_audit.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.research.timestamp_audit import run_timestamp_audit
from src.utils.helpers import get_logger

logger = get_logger("run_research_timestamp_audit")

_SAVE_DIR = "results/research/timestamp_audit"


def main():
    logger.info("=" * 60)
    logger.info("LRRS RESEARCH — Authoritative Feature Timestamp Audit")
    logger.info("Primary source: LCDataDictionary.csv")
    logger.info("=" * 60)

    result = run_timestamp_audit(save_dir=_SAVE_DIR)

    summary = result["summary"]
    verdict = result["temporal_reconciliation"]

    logger.info("\nAudit complete.")
    logger.info("Features classified: %d", len(result["features"]))
    logger.info("\nTiming distribution:")
    for tv, count in sorted(summary["timing_value_counts"].items(), key=lambda x: -x[1]):
        logger.info("  %-40s %d", tv, count)

    logger.info("\nConfirmed post-origination features (LEAKAGE RISK):")
    for feat in summary["confirmed_post_origination_features"]:
        rec = result["features"][feat]
        logger.info("  %-35s risk=%s", feat, rec["leakage_risk"])

    logger.info("\nRECONCILED VERDICT (timing):")
    logger.info("  %s", verdict["reconciled_verdict_timing"])
    logger.info("\nRECONCILED VERDICT (leakage):")
    logger.info("  %s", verdict["reconciled_verdict_leakage"])

    logger.info("\nOutputs:")
    logger.info("  %s/timestamp_audit.json", _SAVE_DIR)
    logger.info("  %s/timestamp_audit_report.txt", _SAVE_DIR)
    logger.info("  %s/timestamp_audit_sources.txt", _SAVE_DIR)


if __name__ == "__main__":
    main()
