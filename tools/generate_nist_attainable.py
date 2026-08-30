#!/usr/bin/env python3
"""
Record how accurate a float64 implementation can *possibly* be on each NIST
univariate dataset.

Some of the certified values cannot be reached in double precision at all.
NumAcc4 holds values like 10000000.2, which is not representable in binary:
storing it already costs 7.5e-10, and since the deviations about the mean are
only ~0.1, that representation error alone caps the standard deviation at
about 8 correct digits. No algorithm recovers digits the input never had.

Rather than hard-coding a lowered threshold and hoping it was lowered for the
right reason, this script measures the ceiling with numpy -- an independent
float64 implementation -- and the test suite then requires Vegoia to match it.
The assertion becomes "as accurate as double precision allows", which stays
honest if someone later makes the PHP code worse.

Usage:  python3 tools/generate_nist_attainable.py
"""

from __future__ import annotations

import json
import math
import pathlib
import re

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
UNIV = ROOT / "resources/fixtures/nist/univ"

DATASETS = ["PiDigits", "Lottery", "Lew", "Mavro", "Michelso",
            "NumAcc1", "NumAcc2", "NumAcc3", "NumAcc4"]


def number(line: str) -> float:
    cleaned = line.replace("(exact)", "").strip()
    return float(re.search(r"(-?\d+\.?\d*(?:[Ee][-+]?\d+)?)\s*$", cleaned).group(1))


def load(name: str):
    lines = (UNIV / f"{name}.dat").read_text().splitlines()
    head = "\n".join(lines[:10])
    cf, ct = map(int, re.search(r"Certified Values:\s*lines\s+(\d+)\s+to\s+(\d+)", head).groups())
    df, dt = map(int, re.search(r"Data\s*:\s*lines\s+(\d+)\s+to\s+(\d+)", head).groups())
    certified = [number(l) for l in lines[cf - 1:ct]]
    data = np.array([number(l) for l in lines[df - 1:dt]], dtype=np.float64)
    return certified, data


def lre(computed: float, certified: float) -> float | None:
    """None encodes an exact hit, which JSON cannot spell as infinity."""
    if computed == certified:
        return None
    denominator = abs(certified) if certified != 0 else 1.0
    return -math.log10(abs(computed - certified) / denominator)


def main() -> int:
    out = {}
    for name in DATASETS:
        (cert_mean, cert_sd, cert_r1), x = load(name)
        deviations = x - np.mean(x)
        out[name] = {
            "mean": lre(float(np.mean(x)), cert_mean),
            "stdDev": lre(float(np.std(x, ddof=1)), cert_sd),
            "autocorrelation": lre(
                float(np.sum(deviations[:-1] * deviations[1:]) / np.sum(deviations ** 2)),
                cert_r1,
            ),
        }

    document = {
        "generator": f"numpy {np.__version__}, IEEE-754 binary64",
        "meaning": (
            "Log Relative Error an independent float64 implementation attains "
            "on each certified value. null means it was hit exactly. Vegoia is "
            "required to match these, so the thresholds track the limits of the "
            "arithmetic rather than the limits of our patience."
        ),
        "regenerate": "python3 tools/generate_nist_attainable.py",
        "datasets": out,
    }

    path = UNIV.parent / "attainable.json"
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")

    print(f"{'dataset':10s} {'mean':>10s} {'stdDev':>10s} {'r(1)':>10s}")
    for name, v in out.items():
        fmt = lambda z: "exact" if z is None else f"{z:.2f}"
        print(f"{name:10s} {fmt(v['mean']):>10s} {fmt(v['stdDev']):>10s} {fmt(v['autocorrelation']):>10s}")
    print(f"\n-> {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
