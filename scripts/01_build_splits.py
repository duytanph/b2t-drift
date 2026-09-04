"""CLI: build and write the temporal train/val/eval session split manifests.

Thin wrapper only -- parses the data config, calls into
`b2t_drift.data.splits`, writes and validates each manifest, and prints a
summary. No split-construction logic lives here.

Generates five manifests: the Switchboard-only cut-point sweep (sessions
1-5, 1-10, 1-15, 1-20), plus one all-corpora fallback at the 1-15 cut in
case Switchboard-only proves too small to train a usable decoder
(`docs/DESIGN.md`, Switchboard-only training decision) -- the specific set
of five is itself the deliberate, concrete design decision this project
made (`docs/DESIGN.md` section 7), not a general parameterization, so it
is spelled out here rather than driven by a config loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from b2t_drift.config import load_yaml
from b2t_drift.data.splits import (
    build_temporal_split,
    load_corpus_labels,
    validate_split,
    write_split,
)

DEFAULT_CONFIG_PATH = Path("configs/data/t15.yaml")

# (cut_session_index, corpus_filter) for each manifest to build, in order.
MANIFEST_SPECS: list[tuple[int, str | None]] = [
    (5, "Switchboard"),
    (10, "Switchboard"),
    (15, "Switchboard"),
    (20, "Switchboard"),
    (15, None),  # all-corpora fallback
]


def main() -> None:
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
    corpus_csv = Path(config["corpus_csv_path"])
    splits_dir = Path(config["splits_dir"])
    corpus_labels = load_corpus_labels(corpus_csv)

    rows = []
    for cut_session_index, corpus_filter in MANIFEST_SPECS:
        split = build_temporal_split(
            data_root, corpus_csv, cut_session_index, corpus_filter=corpus_filter,
        )
        # allow_empty_eval_sessions=True: a zero-trial eval session under
        # corpus filtering is the already-established, deliberate design
        # (docs/DESIGN.md, corpus stratification decision) -- such
        # sessions are kept in the manifest as flagged (passes_threshold:
        # false) entries, not dropped, so their absence of Switchboard
        # data stays auditable rather than silently disappearing.
        validate_split(split, corpus_labels=corpus_labels, data_root=data_root,
                        allow_empty_eval_sessions=True)
        write_split(split, splits_dir / f"{split['name']}.json")

        n_passing = sum(1 for e in split["eval_sessions"] if e["passes_threshold"])
        offsets = [e["days_since_cut"] for e in split["eval_sessions"] if e["passes_threshold"]]
        offset_range = f"{min(offsets)}-{max(offsets)}" if offsets else "n/a"
        rows.append({
            "name": split["name"],
            "cut_date": split["cut_date"],
            "n_train": len(split["train"]),
            "n_val": len(split["val"]),
            "n_eval_sessions": len(split["eval_sessions"]),
            "n_passing": n_passing,
            "offset_range": offset_range,
        })
        print(f"Wrote and validated {splits_dir / (split['name'] + '.json')}")

    print("\n=== Summary ===")
    header = (
        f"{'cut point':<32}{'cut date':<12}{'train trials':<14}{'val trials':<12}"
        f"{'eval sessions':<15}{'passing threshold':<20}{'day-offset range':<18}"
    )
    print(header)
    for row in rows:
        print(
            f"{row['name']:<32}{row['cut_date']:<12}{row['n_train']:<14}{row['n_val']:<12}"
            f"{row['n_eval_sessions']:<15}{row['n_passing']:<20}{row['offset_range']:<18}"
        )


if __name__ == "__main__":
    main()
