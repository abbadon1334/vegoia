#!/usr/bin/env python3
"""
An expected variance computed in exact arithmetic.

The NIST univariate datasets top out at 5000 observations, which is not enough
to exercise the residual correction in the two-pass variance: with the mean
itself accumulated under Neumaier compensation, the residual sum(x - mean) only
grows large enough to matter across hundreds of thousands of values.

Mutation testing found the gap -- deleting the correction left every test
green. This fixture closes it. The sample is 200000 values clustered within a
micro-unit of 1e9, where dropping the correction moves the variance in its
fourth significant digit, and the expected value is computed with Fraction, so
it is exact rather than merely better.

Usage: python3 tools/generate_variance_stress.py
"""

from __future__ import annotations

import decimal
import json
import pathlib
from fractions import Fraction

N = 200_000
BASE = 10 ** 9


def main() -> int:
    values = [float(BASE + (i % 7) * 1e-6) for i in range(N)]
    exact = [Fraction(v) for v in values]

    mean = sum(exact) / N
    variance = sum((v - mean) ** 2 for v in exact) / (N - 1)

    decimal.getcontext().prec = 60
    sd = decimal.Decimal(variance.numerator).sqrt() / decimal.Decimal(variance.denominator).sqrt()

    doc = {
        "note": (
            "Constructed to exercise the Chan-Golub-LeVeque residual correction, which "
            "the NIST univariate sets are too small to reach. Expected values are exact: "
            "computed in rational arithmetic, not in floating point."
        ),
        "generator": "python3 tools/generate_variance_stress.py",
        "count": N,
        "base": BASE,
        "pattern": "base + (i % 7) * 1e-6",
        "mean": float(mean),
        "variance": float(variance),
        "standard_deviation": float(sd),
    }

    pathlib.Path("resources/fixtures/variance_stress.json").write_text(
        json.dumps(doc, indent=1) + "\n"
    )
    print(f"n={N} mean={float(mean):.17g} variance={float(variance):.17g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
