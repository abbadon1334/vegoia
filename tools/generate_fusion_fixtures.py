#!/usr/bin/env python3
"""
Reference values for Reciprocal Rank Fusion, computed as exact rationals.

Unlike everything else in this directory there is no library to generate
against. RRF has no canonical implementation: LangChain, Elasticsearch and
Weaviate each read the conventions differently -- whether ranks start at 0 or
1, whether a document missing from a ranking is charged a rank or contributes
nothing, whether duplicates within one ranking count twice. Generating from any
of them would pin that library's reading rather than the paper's.

So the scores are computed with `fractions.Fraction` and rounded once, at the
end. That is a stronger reference than any implementation and it is the same
move `generate_special_function_fixtures.py` makes with mpmath: when the truth
is available exactly, use the truth.

The formula is Cormack, Clarke and Buettcher (2009):

    score(d) = sum over rankings of 1 / (k + rank(d)),  rank starting at 1

with k = 60, which they chose on TREC data. A document absent from a ranking
contributes nothing.

Usage: python3 tools/generate_fusion_fixtures.py
"""

from __future__ import annotations

import json
import pathlib
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parent.parent

CASES = {
    "one_ranking": {"rankings": [["a", "b", "c"]], "k": 60},
    "two_agreeing": {"rankings": [["a", "b", "c"], ["a", "b", "c"]], "k": 60},
    "two_disagreeing": {"rankings": [["a", "b", "c"], ["c", "b", "a"]], "k": 60},
    # The case the method exists for: a document nobody ranks first, but that
    # everybody ranks highly, beating one that a single engine loved.
    "consensus_beats_enthusiasm": {
        "rankings": [["x", "b", "c"], ["b", "y", "c"], ["b", "c", "z"]],
        "k": 60,
    },
    "partial_overlap": {
        "rankings": [["a", "b"], ["c", "d"], ["b", "d", "a"]],
        "k": 60,
    },
    "duplicate_within_one_ranking": {"rankings": [["a", "b", "a"], ["b", "a"]], "k": 60},
    "one_empty_ranking": {"rankings": [["a", "b"], [], ["b", "a"]], "k": 60},
    # k = 0 is the pure reciprocal rank, where the first place is worth
    # everything; large k flattens towards a plain vote.
    "k_zero": {"rankings": [["a", "b", "c"], ["b", "c", "a"]], "k": 0},
    "k_large": {"rankings": [["a", "b", "c"], ["b", "c", "a"]], "k": 1000},
    "integer_keys": {"rankings": [[10, 20, 30], [30, 10]], "k": 60},
    # Same multiset of ranks reached in different orders: the scores must be
    # identical, which they are only if the summation order is canonical.
    "same_ranks_different_order": {
        "rankings": [["a", "b", "p"], ["b", "a", "q"], ["p", "q", "a"]],
        "k": 60,
    },
}


def fuse(rankings: list[list], k: int) -> dict:
    """Exact scores, then one rounding at the end."""
    ranks: dict = {}

    for ranking in rankings:
        seen = set()

        for position, key in enumerate(ranking, start=1):
            # The first occurrence wins: a document listed twice by one
            # retriever would otherwise collect double credit from a single
            # opinion.
            if key in seen:
                continue

            seen.add(key)
            ranks.setdefault(key, []).append(position)

    scores = {}

    for key, positions in ranks.items():
        total = Fraction(0)

        for rank in sorted(positions, reverse=True):
            total += Fraction(1, k + rank)

        scores[key] = total

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))

    return {
        "keys": [kv[0] for kv in ordered],
        "scores": [float(kv[1]) for kv in ordered],
        "exact": [f"{kv[1].numerator}/{kv[1].denominator}" for kv in ordered],
    }


def main() -> int:
    out = {}

    for name, case in CASES.items():
        out[name] = {**case, "fused": fuse(case["rankings"], case["k"])}

    document = {
        "generator": "exact rational arithmetic (fractions.Fraction), rounded once at the end",
        "regenerate": "python3 tools/generate_fusion_fixtures.py",
        "note": (
            "There is no canonical implementation to generate against -- LangChain, "
            "Elasticsearch and Weaviate read the conventions differently -- so the truth is "
            "computed exactly instead. Ranks start at 1; a document absent from a ranking "
            "contributes nothing; the first occurrence within one ranking wins; ties break on "
            "the key ascending, matching NearestNeighbours."
        ),
        "cases": out,
    }

    path = ROOT / "resources/fixtures/rag/fusion.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")

    print(f"-> {path}")

    for name, case in out.items():
        print(f"   {name:30s} k={case['k']:4d}  {case['fused']['keys']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
