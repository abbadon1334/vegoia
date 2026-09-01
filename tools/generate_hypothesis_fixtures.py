#!/usr/bin/env python3
"""
Reference values for multiple testing correction and the hypothesis tests.

A p-value on its own is a claim about one comparison. Twenty of them at the
five per cent level is a claim about twenty, and one of those is expected to
be wrong by chance alone -- which is the most common mistake in applied
statistics and the one a library that hands out p-values invites you to make.
The corrections here are what turn a family of p-values back into something
that can be read.

statsmodels and SciPy are the references. Where a convention has more than one
defensible reading the generator pins it explicitly rather than accepting a
default, and says so in the note, because a default is a decision somebody
else made and it can change under you.

Usage: python3 tools/generate_hypothesis_fixtures.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = pathlib.Path(__file__).resolve().parent.parent

METHODS = {"bonferroni": "bonferroni", "holm": "holm", "benjamini_hochberg": "fdr_bh"}

# Chosen for the shapes that separate the three procedures rather than for
# variety. Holm equals Bonferroni on the smallest p-value and diverges after
# it; Benjamini-Hochberg's step-up is only visible when the raw ratios are not
# already monotone; and both step procedures have to survive an input that is
# not sorted, since nothing says a caller's p-values arrive in order.
FAMILIES = {
    "evenly_spaced": [0.01, 0.02, 0.03, 0.04, 0.05],
    "unsorted": [0.05, 0.01, 0.04, 0.02, 0.03],
    "one_tiny_one_huge": [0.001, 0.9, 0.02, 0.03],
    # The step-up earns its keep here: the raw BH ratios are 0.08 and 0.041,
    # which are decreasing, so thresholding them without the monotone pass
    # would reject the larger p-value and not the smaller.
    "ratios_out_of_order": [0.04, 0.041],
    "with_ties": [0.01, 0.01, 0.03, 0.9],
    "all_significant": [0.001, 0.002, 0.003],
    "none_significant": [0.4, 0.5, 0.6, 0.7],
    "single": [0.02],
    "at_the_boundary": [0.0, 1.0, 0.5],
    "twenty_uniform": [round(0.05 * i, 10) for i in range(1, 21)],
}


# --- the four tests ---------------------------------------------------------

CONTINGENCY = {
    # 2x2 is the only shape Yates touches, so both readings are recorded for
    # every one of them and only the corrected reading differs.
    "two_by_two": [[10, 20], [30, 25]],
    "two_by_two_balanced": [[25, 25], [25, 25]],
    # |observed - expected| is 0.244 here, below the half-step Yates would
    # subtract. The textbook formula overshoots and reports 0.0256 where the
    # answer is zero; SciPy clamps the shift to the difference and gets it.
    "two_by_two_under_half_a_step": [[10, 10], [10, 11]],
    "three_by_three": [[10, 20, 30], [30, 25, 15], [5, 10, 20]],
    "three_by_four": [[10, 20, 30, 15], [30, 25, 15, 20], [5, 10, 20, 25]],
    "large_counts": [[1000, 2000], [3000, 2500]],
}

SAMPLES = {
    "unequal_variance_and_size": (
        [5.1, 4.9, 6.2, 5.8, 5.5, 5.0, 6.1, 5.9],
        [4.2, 4.8, 4.5, 5.1, 4.4, 4.9, 4.7, 4.3, 5.0],
    ),
    "equal_variance": ([1.0, 2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0, 7.0]),
    "wildly_unequal_spread": ([1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0]),
    "identical_samples": ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
    "with_ties": ([1.0, 1.0, 2.0, 3.0], [2.0, 3.0, 3.0, 4.0]),
    "heavily_tied": ([1.0, 2.0, 2.0, 3.0, 4.0], [2.0, 3.0, 3.0, 5.0, 6.0]),
    "no_overlap": ([1.0, 2.0, 3.0], [10.0, 11.0, 12.0]),
    "large_and_shifted": (
        [1e8 + x for x in (1.0, 2.0, 3.0, 4.0, 5.0)],
        [1e8 + x for x in (3.0, 4.0, 5.0, 6.0, 7.0)],
    ),
}

GROUPS = {
    "three_clean": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.5]],
    "three_with_ties": [[1.0, 1.0, 2.0, 3.0], [2.0, 2.0, 3.0, 4.0], [3.0, 3.0, 4.0, 5.0]],
    "two_groups": [[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]],
    "uneven_sizes": [[1.0, 2.0], [3.0, 4.0, 5.0, 6.0, 7.0], [8.0, 9.0]],
    "no_difference": [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
    "many_groups": [[float(i + j) for j in range(4)] for i in range(6)],
}


def chi_squared_section() -> dict:
    out = {}

    for name, table in CONTINGENCY.items():
        entry = {"table": table}

        for label, correction in (("corrected", True), ("uncorrected", False)):
            r = stats.chi2_contingency(np.array(table, dtype=float), correction=correction)
            entry[label] = {
                "statistic": float(r.statistic),
                "p_value": float(r.pvalue),
                "degrees_of_freedom": int(r.dof),
            }

        entry["expected"] = [[float(v) for v in row]
                             for row in stats.chi2_contingency(np.array(table, dtype=float)).expected_freq]
        out[name] = entry

    return out


def t_section() -> dict:
    out = {}

    for name, (x, y) in SAMPLES.items():
        entry = {"x": x, "y": y}

        for label, equal in (("student", True), ("welch", False)):
            r = stats.ttest_ind(x, y, equal_var=equal)
            low, high = r.confidence_interval(confidence_level=0.95)
            entry[label] = {
                "statistic": float(r.statistic),
                "p_value": float(r.pvalue),
                "degrees_of_freedom": float(r.df),
                "confidence_interval": [float(low), float(high)],
            }

        out[name] = entry

    return out


def mann_whitney_section() -> dict:
    """Always method='asymptotic', never 'auto'.

    SciPy's 'auto' switches between the exact and the asymptotic route on
    sample size and on whether ties are present, and the two give materially
    different answers -- on one of the samples here, 0.1138 against 0.1508. A
    fixture generated with 'auto' would therefore record which branch SciPy
    took on that input, which is a property of its heuristic rather than of the
    mathematics, and changing one observation would silently change procedure
    while the file looked the same.
    """
    out = {}

    for name, (x, y) in SAMPLES.items():
        entry = {"x": x, "y": y}

        for alt in ("two-sided", "less", "greater"):
            for label, cc in (("corrected", True), ("uncorrected", False)):
                r = stats.mannwhitneyu(
                    x, y, use_continuity=cc, alternative=alt, method="asymptotic"
                )
                entry[f"{alt}_{label}"] = {
                    "statistic": float(r.statistic),
                    "p_value": float(r.pvalue),
                }

        out[name] = entry

    return out


def kruskal_section() -> dict:
    out = {}

    for name, groups in GROUPS.items():
        r = stats.kruskal(*groups)
        out[name] = {
            "groups": groups,
            "statistic": float(r.statistic),
            "p_value": float(r.pvalue),
            "degrees_of_freedom": len(groups) - 1,
        }

    return out


def main() -> int:
    families = {}

    for name, p in FAMILIES.items():
        entry = {"p_values": p}

        for label, method in METHODS.items():
            reject, adjusted, _, _ = multipletests(p, alpha=0.05, method=method)
            entry[label] = {
                "adjusted": [float(v) for v in adjusted],
                "rejected": [bool(v) for v in reject],
            }

        families[name] = entry

    document = {
        "generator": "statsmodels.stats.multitest.multipletests, alpha=0.05",
        "regenerate": "python3 tools/generate_hypothesis_fixtures.py",
        "note": (
            "Adjusted p-values are returned in input order, under the input positions, and are "
            "clamped to 1. Holm is step-down and Benjamini-Hochberg is step-up; both enforce "
            "monotonicity across the sorted family, which is why 'ratios_out_of_order' is here. "
            "Rejection is adjusted <= alpha. "
            "Chi-squared is recorded both with and without Yates' correction, which SciPy applies "
            "by default and which only touches 2x2 tables. Mann-Whitney is generated with "
            "method='asymptotic' passed explicitly, never 'auto': auto switches route on sample "
            "size and on the presence of ties, and the two routes disagree, so a fixture built "
            "with it would pin SciPy's heuristic rather than the mathematics. The H that SciPy "
            "reports for Kruskal-Wallis is already divided by the tie-correction factor."
        ),
        "alpha": 0.05,
        "multiple_testing": families,
        "chi_squared": chi_squared_section(),
        "t": t_section(),
        "mann_whitney": mann_whitney_section(),
        "kruskal_wallis": kruskal_section(),
    }

    path = ROOT / "resources/fixtures/stats/hypothesis.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")

    print(f"-> {path}")
    print(f"   {len(families)} families")

    for section in ("chi_squared", "t", "mann_whitney", "kruskal_wallis"):
        print(f"   {len(document[section]):2d} {section}")

    print(f"   {len(families)} families of p-values")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
