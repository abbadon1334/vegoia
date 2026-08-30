#!/usr/bin/env python3
"""
Reference values for the partition-agreement measures, from scikit-learn.

NMI and ARI both have several conventions in circulation -- NMI can normalise
by the arithmetic mean, the geometric mean, the minimum or the maximum of the
two entropies, and they disagree. Pinning against sklearn's defaults keeps our
numbers comparable with what anyone else would compute.

The cases include the degenerate ones on purpose: identical partitions, one
community, all singletons, and total disagreement. Those are where the formulas
divide by zero and where implementations quietly differ.

Usage: python3 tools/generate_agreement_fixtures.py
"""

from __future__ import annotations

import json
import pathlib

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

CASES = {
    "identical": ([0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2]),
    "relabelled": ([0, 0, 1, 1, 2, 2], [5, 5, 9, 9, 1, 1]),
    "one_pair_moved": ([0, 0, 1, 1, 2, 2], [0, 0, 1, 2, 2, 2]),
    "split_in_two": ([0, 0, 0, 0], [0, 0, 1, 1]),
    "merged": ([0, 0, 1, 1], [0, 0, 0, 0]),
    "all_singletons_vs_one": ([0, 1, 2, 3], [0, 0, 0, 0]),
    "singletons_vs_singletons": ([0, 1, 2, 3], [3, 2, 1, 0]),
    "single_community_both": ([0, 0, 0], [0, 0, 0]),
    "orthogonal": ([0, 0, 1, 1], [0, 1, 0, 1]),
    "one_node": ([0], [0]),
    "uneven": ([0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 1, 1]),
}


def main() -> int:
    out = {}
    for name, (a, b) in CASES.items():
        out[name] = {
            "a": a,
            "b": b,
            "nmi": normalized_mutual_info_score(a, b),          # arithmetic mean
            "ari": adjusted_rand_score(a, b),
        }
        print(f"{name:26s} NMI={out[name]['nmi']:.15f}  ARI={out[name]['ari']:.15f}")

    doc = {
        "generator": "scikit-learn, normalized_mutual_info_score(average_method='arithmetic') and adjusted_rand_score",
        "regenerate": "python3 tools/generate_agreement_fixtures.py",
        "cases": out,
    }
    path = pathlib.Path("resources/fixtures/partition_agreement.json")
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
