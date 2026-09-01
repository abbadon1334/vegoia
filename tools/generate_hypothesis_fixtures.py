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
            "Rejection is adjusted <= alpha."
        ),
        "alpha": 0.05,
        "multiple_testing": families,
    }

    path = ROOT / "resources/fixtures/stats/hypothesis.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")

    print(f"-> {path}")
    print(f"   {len(families)} families")

    for name, entry in families.items():
        bh = entry["benjamini_hochberg"]["adjusted"]
        print(f"     {name:22s} n={len(entry['p_values']):2d}  BH -> "
              f"{[round(v, 6) for v in bh[:4]]}{' ...' if len(bh) > 4 else ''}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
