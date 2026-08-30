"""CLI: sanity-check that the raw T15 data is present and readable.

Thin wrapper only -- parses arguments/config and calls into
`b2t_drift.data.hdf5_io`. No logic belongs here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from b2t_drift.config import load_yaml
from b2t_drift.data.hdf5_io import (
    build_session_manifest,
    session_date_gaps,
    write_session_manifest,
)

DEFAULT_CONFIG_PATH = Path("configs/data/t15.yaml")


def main() -> None:
    """Entry point: verify raw data files, write the manifest, print a report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the data config YAML (default: %(default)s).",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    data_root = Path(config["raw_data_dir"])
    manifest_path = Path(config["manifest_path"])

    result = build_session_manifest(data_root)
    write_session_manifest(result.rows, manifest_path)

    total_trials = sum(row["n_trials"] for row in result.rows)
    print(f"Wrote {len(result.rows)} manifest rows to {manifest_path}")
    print(f"Total trials across all sessions: {total_trials}")

    per_partition_totals: dict[str, int] = {}
    for row in result.rows:
        per_partition_totals[row["partition"]] = (
            per_partition_totals.get(row["partition"], 0) + row["n_trials"]
        )
    print("Per-partition totals:")
    for partition in ("train", "val", "test"):
        print(f"  {partition}: {per_partition_totals.get(partition, 0)}")

    if result.unexpected_partition_patterns:
        print("\nSessions with an unexpected partition-file pattern:")
        for session_id, present in result.unexpected_partition_patterns:
            print(f"  {session_id}: has {sorted(present)}")
    else:
        print("\nNo sessions with an unexpected partition-file pattern.")

    if result.issues:
        print(f"\n{len(result.issues)} trial(s) missing an expected attribute:")
        for issue in result.issues[:20]:
            print(
                f"  {issue.session_id}/{issue.partition}/{issue.trial_key}: "
                f"missing '{issue.missing_attr}'"
            )
        if len(result.issues) > 20:
            print(f"  ... and {len(result.issues) - 20} more")
    else:
        print("\nNo trials missing an expected attribute.")

    print("\nSession dates and day-gaps between consecutive sessions:")
    for session_id, session_date, gap_days in session_date_gaps(result.sessions):
        gap_str = "" if gap_days is None else f"  [+{gap_days}d]"
        print(f"  {session_id}  {session_date.isoformat()}{gap_str}")


if __name__ == "__main__":
    main()
