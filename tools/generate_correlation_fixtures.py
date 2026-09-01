#!/usr/bin/env python3
"""
Reference values for the correlation coefficients, from SciPy.

The three measure different things and are not interchangeable, which is why
all three are implemented and pinned:

  * Pearson measures linear association, and only that. It is 0 for a perfect
    parabola.
  * Spearman is Pearson on the ranks, so it captures any monotone
    relationship, linear or not.
  * Kendall counts concordant against discordant pairs, which makes it robust
    to outliers where Spearman is only resistant.

Tie handling is where implementations quietly diverge. Spearman needs midranks;
Kendall has three tie corrections in circulation (tau-a, tau-b, tau-c) and
SciPy's default is tau-b, which is what is pinned here. The cases below include
ties on purpose.

Usage: python3 tools/generate_correlation_fixtures.py
"""

from __future__ import annotations

import json
import pathlib

from scipy import stats

CASES = {
    "perfect_positive": ([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]),
    "perfect_negative": ([1, 2, 3, 4, 5], [10, 8, 6, 4, 2]),
    "monotone_not_linear": ([1, 2, 3, 4, 5], [1, 4, 9, 16, 25]),
    "parabola": ([-2, -1, 0, 1, 2], [4, 1, 0, 1, 4]),
    "noisy": ([1, 2, 3, 4, 5, 6, 7, 8], [2, 1, 4, 3, 6, 5, 8, 7]),
    "with_ties_x": ([1, 1, 2, 2, 3, 3], [1, 2, 3, 4, 5, 6]),
    # Two values that are distinct as doubles and identical once PHP casts
    # them to string at its default precision of 14. Nothing about the data is
    # unusual -- 0.1 and the next few ulps above it -- but a tie test written
    # on the string form counts them as tied while a tie test written on the
    # value does not, and Kendall's tau-b needs both halves to agree.
    "ties_only_after_rounding": (
        [0.1, 0.10000000000000012, 0.2, 0.3, 0.4, 0.5],
        [1.0, 2.0, 3.0, 6.0, 4.0, 5.0],
    ),
    "with_ties_both": ([1, 1, 2, 2, 3, 3], [1, 1, 2, 2, 3, 3]),
    "anscombe_1": (
        [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
    ),
    "anscombe_2": (
        [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74],
    ),
    "anscombe_4": (
        [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
        [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89],
    ),
    "large_values": ([1e8 + 1, 1e8 + 2, 1e8 + 3, 1e8 + 4], [1e8 + 2, 1e8 + 4, 1e8 + 6, 1e8 + 8]),
}


def main() -> int:
    out = {}
    for name, (x, y) in CASES.items():
        out[name] = {
            "x": [float(v) for v in x],
            "y": [float(v) for v in y],
            "pearson": float(stats.pearsonr(x, y).statistic),
            "spearman": float(stats.spearmanr(x, y).statistic),
            "kendall": float(stats.kendalltau(x, y).statistic),   # tau-b
        }
        print(f"{name:22s} r={out[name]['pearson']:9.6f}  rho={out[name]['spearman']:9.6f}  "
              f"tau={out[name]['kendall']:9.6f}")

    doc = {
        "generator": "scipy.stats pearsonr / spearmanr / kendalltau (tau-b)",
        "regenerate": "python3 tools/generate_correlation_fixtures.py",
        "cases": out,
    }
    path = pathlib.Path("resources/fixtures/correlation.json")
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
