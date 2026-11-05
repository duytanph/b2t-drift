"""Standalone diagnostic: are the three confirmed measured changes actually
inter-correlated with EACH OTHER, or only each separately correlated with
the PER curve (which is itself monotone in time, making correlation among
time-monotone series close to guaranteed regardless of a shared cause)?

docs/DESIGN.md section 6 states the three confirmed measured changes
(SNR loss, tuning change, subspace rotation) are "inter-correlated at
|rho|=0.70-0.90." No FINDINGS.md entry actually computed a pairwise
inter-measure correlation under that framing -- the number was built from
three SEPARATE entries, each reporting one measured change's own Spearman
correlation AGAINST THE s05 PER CURVE (SNR/TX: rho=-0.750; tuning:
rho=-0.898 to -0.903; rotation: rho=0.908), not against each other. This
tool computes what was actually implied (raw pairwise correlation between
the three measured changes' own per-session values) and, since all three trend
with elapsed time, what matters more: the SAME pairwise correlations with
days-since-cut partialled out, which is the only way to tell whether the
three move together beyond their shared time trend. See docs/FINDINGS.md:
"Raw cross-measure correlation is confounded by a shared time trend; it
disappears once days-since-cut is partialled out" for the investigation
this script resulted from.

DATA PROVENANCE, STATED PLAINLY. This reads
scratch/task_f2_summary.json by default -- itself a byproduct of
scratch/task_f_attribution_verdict.py, never committed (scratch/ is
gitignored). That script derives its three per-session series from other
gitignored scratch/ caches (scratch/tuning_matrices/*.npz,
scratch/task_d_summary.json, scratch/task_d_summary2.json), which in turn
require the pretrained checkpoint's CTC forward-backward pass and a 512x512
PCA per session to regenerate -- genuinely expensive to reproduce from
scratch, unlike this tool's own computation (pure numpy/scipy on cached
arrays, seconds). This is the SAME "committed claim, no committed source"
problem already fixed twice this project for other figures; it is NOT
fixed here -- flagged, not solved, since regenerating the underlying
per-session measures is out of this tool's scope. If scratch/
task_f2_summary.json is ever deleted, this tool stops working until those
upstream scratch/ scripts are re-run.

Run with: uv run tools/attribution_partial_correlation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

DEFAULT_INPUT = Path("scratch/task_f2_summary.json")

# (display name, measure-A key, measure-B key) -- the three pairs among
# the project's three CONFIRMED attribution measured changes. "tx_norm" (SNR loss)
# and "corr_all" (tuning change) both DECLINE with drift; "angle_values"
# (subspace rotation) INCREASES with drift -- signs below are reported as
# measured, not flipped to a common convention.
PAIRS = (
    ("SNR (tx_norm) vs tuning (corr_all)", "tx_norm", "corr_all"),
    ("SNR (tx_norm) vs rotation (angle_values)", "tx_norm", "angle_values"),
    ("tuning (corr_all) vs rotation (angle_values)", "corr_all", "angle_values"),
)


def _fisher_ci(rho: float, n: int, k: int, alpha: float) -> tuple[float, float]:
    """95%-style Fisher-z confidence interval for a Spearman rho.

    Args:
        rho: The (raw or partial) correlation.
        n: Sample size.
        k: Number of variables partialled out (0 for raw, 1 for the
            single-control partial correlations this tool computes).
        alpha: Two-sided significance level (0.05 for a 95% CI).

    Returns:
        `(lower, upper)` bounds on rho.
    """
    z = np.arctanh(rho)
    se = 1.0 / np.sqrt(n - 3 - k)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - zcrit * se)), float(np.tanh(z + zcrit * se))


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Spearman partial correlation of x and y, controlling for z.

    Implemented by rank-transforming all three series, regressing ranked x
    and ranked y each on ranked z (ordinary least squares with intercept),
    and Pearson-correlating the two residual sets -- the "regress on days,
    then correlate the residuals" construction, applied to ranks so it is
    directly comparable to the RAW Spearman correlations this tool also
    reports (same correlation type on both sides, per the investigation
    this tool exists for). Cross-checked against the standard partial-
    correlation formula applied to the three pairwise Spearman rhos
    (`(rho_xy - rho_xz*rho_yz) / sqrt((1-rho_xz^2)(1-rho_yz^2))`); the two
    methods agree to within floating-point tolerance.

    Args:
        x: First series.
        y: Second series.
        z: Series to control for (days-since-cut, here).

    Returns:
        `(partial_rho, p_value)`. The p-value is the standard Pearson
        correlation significance test on the residuals, with the usual
        caveat that it assumes the residuals are approximately normal --
        adequate for n=27, not claimed to be exact.
    """
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)

    def _residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        design = np.vstack([b, np.ones_like(b)]).T
        coef, *_ = np.linalg.lstsq(design, a, rcond=None)
        return a - design @ coef

    r, p = stats.pearsonr(_residual(rx, rz), _residual(ry, rz))
    return float(r), float(p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                         help=f"Per-session summary JSON (default: {DEFAULT_INPUT}).")
    parser.add_argument("--alpha", type=float, default=0.05,
                         help="Two-sided significance level for confidence intervals (default: 0.05).")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input} not found. This tool reads a cached per-session "
            f"summary, not raw data -- see this script's module docstring "
            f"for what regenerates it if it's missing."
        )
    data = json.loads(args.input.read_text())
    days = np.array(data["days_cut"])
    per = np.array(data["per_values"])
    series = {
        "tx_norm": np.array(data["tx_norm"]),
        "corr_all": np.array(data["corr_all"]),
        "angle_values": np.array(data["angle_values"]),
    }
    n = len(days)

    print(f"n = {n} (matched s05 held-out sessions, days-since-cut {days.min():.0f}-{days.max():.0f})")

    print("\n=== Sanity check: each measured change's own correlation against PER ===")
    for name, s in series.items():
        rho, p = stats.spearmanr(s, per)
        print(f"  {name} vs PER: Spearman rho={rho:.4f}, p={p:.3g}")

    print("\n=== RAW pairwise inter-measure correlations (Spearman) ===")
    for label, key_a, key_b in PAIRS:
        rho, p = stats.spearmanr(series[key_a], series[key_b])
        lo, hi = _fisher_ci(rho, n, k=0, alpha=args.alpha)
        print(f"  {label}: rho={rho:.4f}, p={p:.3g}, "
              f"{100 * (1 - args.alpha):.0f}% CI [{lo:.3f}, {hi:.3f}]")

    print(f"\n=== PARTIAL correlations controlling for days-since-cut (Spearman, n={n}) ===")
    for label, key_a, key_b in PAIRS:
        rho, p = _partial_spearman(series[key_a], series[key_b], days)
        lo, hi = _fisher_ci(rho, n, k=1, alpha=args.alpha)
        print(f"  {label}: partial rho={rho:.4f}, p={p:.3g}, "
              f"{100 * (1 - args.alpha):.0f}% CI [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
