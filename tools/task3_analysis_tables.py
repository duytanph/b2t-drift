"""Phase 3 Task 3 analysis: aggregate results/recalibration_grid.csv into
per-(arm, session, N) summaries and print every table the write-up needs.

INPUT: results/recalibration_grid.csv (committed) -- the same recalibration
grid tools/recovery_figure.py reads, one row per (arm, session, N,
calibration split). No checkpoint, no HDF5 load.

Imported by tools/task3_make_figure.py (figures/recalibration_arms.png);
also runnable directly for its own printed tables.

Run with: uv run tools/task3_analysis_tables.py
"""

from __future__ import annotations

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

INPUT = Path("results/recalibration_grid.csv")

SESSIONS = ["t15.2023.09.03", "t15.2023.10.06", "t15.2023.10.22", "t15.2024.07.19", "t15.2025.04.13"]
ARMS = ["1a", "1b", "2", "3a", "3b"]
LABELED_ARMS = ["2", "3a", "3b"]
UNLABELED_ARMS = ["1a", "1b"]
NS = [10, 25, 50, 100]

if not INPUT.exists():
    raise SystemExit(
        f"{INPUT} not found. This module needs the committed Arm 1a/1b/2/3a/3b "
        f"recalibration grid (one row per arm/session/N/calibration-split)."
    )

rows = list(csv.DictReader(open(INPUT)))
days = {r["session_id"]: int(r["days_since_cut"]) for r in rows}

# cell[(arm, session, n)] -> dict of lists over splits
cell = defaultdict(lambda: defaultdict(list))
for r in rows:
    key = (r["arm"], r["session_id"], int(r["n_calibration"]))
    cell[key]["per"].append(float(r["per"]))
    cell[key]["arm0"].append(float(r["arm0_per"]))
    # De-duplicated values where they exist (every row except Arm 2's 32
    # leak-affected cells, which were not refit); fall back to the original
    # otherwise, so a consumer that asks for "per_dedup" always gets a value.
    cell[key]["per_dedup"].append(float(r["per_dedup"]) if r.get("per_dedup") else float(r["per"]))
    cell[key]["arm0_dedup"].append(float(r["arm0_per_dedup"]) if r.get("arm0_per_dedup") else float(r["arm0_per"]))
    cell[key]["min"].append(float(r["calibration_minutes"]))
    cell[key]["impr"].append(float(r["per_improvement"]))
    cell[key]["sec"].append(float(r["wall_seconds"]) if r.get("wall_seconds") else 0.0)


TOTAL_TRIALS = {
    r["session_id"]: int(r["n_eval_trials"]) + int(r["n_calib_trials"])
    for r in rows
    if r["arm"] == "0" and r["n_calibration"] == "10" and r["split_index"] == "0"
}


def mean(arm, s, n, f="per"):
    return st.mean(cell[(arm, s, n)][f])


def sd(arm, s, n, f="per"):
    v = cell[(arm, s, n)][f]
    return st.stdev(v) if len(v) > 1 else 0.0


def have(arm, s, n):
    return (arm, s, n) in cell


if __name__ == "__main__":
    print("=== Arm 0 (paired baseline on each cell's own eval partition) and calibration minutes ===")
    print(f"{'session':16} {'days':>5} | " + " | ".join(f"N={n:<3} arm0   min" for n in [0] + NS))
    for s in SESSIONS:
        parts = []
        for n in [0] + NS:
            if have("0", s, n):
                parts.append(f"{mean('0', s, n):.4f}  {mean('0', s, n, 'min'):5.1f}")
            else:
                parts.append("  --- (skipped)")
        print(f"{s:16} {days[s]:>5} | " + " | ".join(parts))

    print("\n=== Mean PER (± sd over 5 splits) per arm / session / N ===")
    for s in SESSIONS:
        print(f"\n{s} (+{days[s]}d)")
        print(f"  {'N':>4} {'min':>6} {'arm0':>8} | " + " ".join(f"{a:>13}" for a in ARMS))
        for n in NS:
            if not have("0", s, n):
                print(f"  {n:>4}  SKIPPED: session has only {TOTAL_TRIALS[s]} labeled trials -> N={n} leaves none to evaluate")
                continue
            line = f"  {n:>4} {mean('0', s, n, 'min'):>6.1f} {mean('0', s, n):>8.4f} | "
            line += " ".join(f"{mean(a, s, n):.4f}±{sd(a, s, n):.3f}" for a in ARMS)
            print(line)

    print("\n=== Recovered PER = Arm 0 - arm (mean over splits; positive = better than no adaptation) ===")
    for s in SESSIONS:
        print(f"\n{s} (+{days[s]}d)")
        print(f"  {'N':>4} " + " ".join(f"{a:>8}" for a in ARMS))
        for n in NS:
            if not have("0", s, n):
                continue
            print(f"  {n:>4} " + " ".join(f"{mean(a, s, n, 'impr'):>+8.4f}" for a in ARMS))

    print("\n=== Best arm per (session, N) by mean PER ===")
    for s in SESSIONS:
        line = f"{s} (+{days[s]:>3}d): "
        bits = []
        for n in NS:
            if not have("0", s, n):
                bits.append(f"N={n}: skipped")
                continue
            best = min(ARMS, key=lambda a: mean(a, s, n))
            base = mean("0", s, n)
            bits.append(f"N={n}: {best} ({mean(best, s, n):.4f} vs arm0 {base:.4f})")
        print(line + " | ".join(bits))

    print("\n=== Crossover: smallest N where each labeled arm beats the best unlabeled arm ===")
    for s in SESSIONS:
        print(f"{s} (+{days[s]}d)")
        for a in LABELED_ARMS:
            found = None
            for n in NS:
                if not have("0", s, n):
                    continue
                best_unl = min(mean(u, s, n) for u in UNLABELED_ARMS)
                if mean(a, s, n) < best_unl:
                    found = n
                    minutes = mean(a, s, n, "min")
                    break
            if found is None:
                print(f"   arm {a}: never beats the best unlabeled arm at any measured N")
            else:
                print(f"   arm {a}: N={found} (~{minutes:.1f} min of calibration)")

    print("\n=== Any (arm, N, session) WORSE than Arm 0: mean level, and count of splits ===")
    for arm in ARMS:
        for s in SESSIONS:
            for n in NS:
                if not have(arm, s, n):
                    continue
                d = cell[(arm, s, n)]
                worse_splits = sum(1 for p, a in zip(d["per"], d["arm0"]) if p > a)
                mean_delta = st.mean(p - a for p, a in zip(d["per"], d["arm0"]))
                if mean_delta > 0 or worse_splits > 0:
                    tag = "MEAN WORSE" if mean_delta > 0 else "some splits worse"
                    print(f"  arm {arm:>2} {s} N={n:>3}: {worse_splits}/5 splits worse, mean delta {mean_delta:+.4f}  [{tag}]")
