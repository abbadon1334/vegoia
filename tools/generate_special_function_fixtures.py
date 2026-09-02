#!/usr/bin/env python3
"""
Reference values for the special functions the distributions are built on.

PHP ships none of these. There is no erf, no erfc, no lgamma, no incomplete
gamma or beta anywhere in the language or in a common extension, which is the
practical reason a PHP library cannot give you a p-value: the whole tower has
to be built first. This file pins the bottom of it.

The method is the one the NIST work in this library already uses, because the
alternative -- checking one implementation against another and calling
agreement correctness -- cannot tell you which of the two is wrong.

  * mpmath at 50 digits gives the certified value. It is arbitrary-precision
    and independent of SciPy, so it plays the role NIST's certified values
    play for the regression datasets.
  * SciPy is then measured against that certified value, and what it reaches
    is recorded as the attainable accuracy. SciPy's special functions are
    Cephes and Boost, the same code that stands behind most of the scientific
    Python stack, so this is a fair account of what a good implementation gets
    on a double.
  * The PHP side is required to come within half a digit of SciPy. A bar set
    that way cannot be met by an implementation that merely looks plausible,
    and cannot demand accuracy that a double cannot hold.

The arguments are chosen for the places these functions are hard, not the
places they are easy: the incomplete gamma where x sits near a and the series
and the continued fraction trade places, the incomplete beta near 0 and 1, and
erfc far enough into the tail that computing it as 1 - erf would have thrown
away every significant digit.

Usage: python3 tools/generate_special_function_fixtures.py
"""

from __future__ import annotations

import json
import math
import pathlib

import mpmath as mp
from scipy import special, stats

mp.mp.dps = 50

# Enough digits that reading the string back as a double is correctly rounded,
# so the certified value stays certified on the PHP side.
DIGITS = 30


def certified(value: mp.mpf) -> str:
    return mp.nstr(value, DIGITS, strip_zeros=False)


# Below this the true value has no double to be stored in, so returning zero
# is the correct answer rather than a loss of accuracy. Measuring digits of
# agreement there would record 0.00 and read like a failure.
UNDERFLOW = mp.mpf("1e-300")


def lre(computed: float, exact: mp.mpf) -> float | None:
    """Log relative error: the digits `computed` agrees with `exact` to.

    None means it agreed exactly, which is the interesting case and must not
    be confused with a missing measurement.
    """
    if computed == exact:
        return None

    error = abs((mp.mpf(computed) - exact) / exact)

    return float(-mp.log10(error))


def case(name: str, exact: mp.mpf, computed: float) -> dict:
    # An exact zero is not an underflow. lgamma(1) and lgamma(2) are zero
    # because gamma is 1 there, and relative error has no meaning at a
    # certified zero -- the PHP side measures those against an absolute error
    # instead, which is the strict reading, not a lenient one.
    if exact != 0 and abs(exact) < UNDERFLOW:
        # The true value is zero, or too small for any double to hold. Pinned as
        # a requirement to return zero, not as an accuracy target.
        return {
            "certified": certified(exact),
            "scipy": float(computed),
            "attainable": None,
            "vanishes": True,
        }

    return {
        "certified": certified(exact),
        "scipy": float(computed),
        "attainable": lre(computed, exact),
    }


def log_gamma() -> dict:
    # 1.46163... is where log-gamma turns; either side of it the function is
    # ill-conditioned in a way a Lanczos series has to survive.
    points = [1e-8, 1e-4, 0.1, 0.5, 1.0, 1.4616321449683623, 1.5, 2.0, 2.5,
              3.5, 7.0, 10.0, 50.0, 100.0, 1000.0, 1e6, 1e8]

    return {
        repr(x): case(f"lgamma({x})", mp.loggamma(mp.mpf(repr(x))), float(special.gammaln(x)))
        for x in points
    }


def error_function() -> dict:
    points = [0.0, 1e-8, 0.05, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0,
              -0.5, -1.0, -3.0]
    out = {}

    for x in points:
        m = mp.mpf(repr(x))
        out[repr(x)] = {
            "erf": case("erf", mp.erf(m), float(special.erf(x))),
            "erfc": case("erfc", mp.erfc(m), float(special.erfc(x))),
        }

    # erfc only: past about x = 6 the value is below 1e-17, so erf is 1 to a
    # double and only the complement carries information. This is exactly the
    # region a normal-tail p-value lives in.
    for x in [6.0, 8.0, 10.0, 15.0, 20.0, 26.0]:
        out[repr(x)] = {"erfc": case("erfc", mp.erfc(mp.mpf(repr(x))), float(special.erfc(x)))}

    return out


def incomplete_gamma() -> dict:
    out = {}

    for a in [0.5, 1.0, 2.0, 3.0, 7.5, 20.0, 100.0, 1000.0]:
        for ratio in [0.01, 0.25, 0.5, 0.9, 1.0, 1.1, 2.0, 5.0]:
            x = a * ratio
            key = f"a={a!r},x={x!r}"
            ma, mx = mp.mpf(repr(a)), mp.mpf(repr(x))
            out[key] = {
                "a": a,
                "x": x,
                # P is the lower tail, Q the upper. Both are pinned because
                # each is the accurate one over half the range, and a library
                # that computes one as 1 minus the other is wrong in the tail
                # it did not choose.
                "P": case("P", mp.gammainc(ma, 0, mx, regularized=True),
                          float(special.gammainc(a, x))),
                "Q": case("Q", mp.gammainc(ma, mx, mp.inf, regularized=True),
                          float(special.gammaincc(a, x))),
            }

    return out


def incomplete_beta() -> dict:
    out = {}

    for a, b in [(0.5, 0.5), (1.0, 1.0), (2.0, 3.0), (5.0, 2.0), (0.1, 0.1),
                 (10.0, 10.0), (50.0, 5.0), (100.0, 100.0), (1000.0, 2.0)]:
        for x in [1e-6, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999999]:
            key = f"a={a!r},b={b!r},x={x!r}"
            out[key] = {
                "a": a,
                "b": b,
                "x": x,
                "I": case("I", mp.betainc(mp.mpf(repr(a)), mp.mpf(repr(b)), 0,
                                          mp.mpf(repr(x)), regularized=True),
                          float(special.betainc(a, b, x))),
            }

    return out


def gamma_prefactor() -> dict:
    """x^a e^-x / gamma(a): the factor in front of both incomplete gammas.

    Public in the PHP library because the chi-squared density is exactly this
    over x, so it is a value callers can land on directly and not only an
    internal step. It is also where the cancellation lives: at a = 1000 the
    numerator and gamma(a) are each astronomically large and the ratio is
    ordinary, so anything that forms the two separately has already lost.

    SciPy's independent route is the gamma density, whose pdf at shape a is
    x^(a-1) e^-x / gamma(a) -- one factor of x away.
    """
    out = {}

    for a in [0.5, 1.0, 2.0, 7.5, 20.0, 100.0, 1000.0]:
        for ratio in [0.01, 0.5, 1.0, 2.0, 10.0]:
            x = a * ratio
            key = f"a={a!r},x={x!r}"
            ma, mx = mp.mpf(repr(a)), mp.mpf(repr(x))
            out[key] = {
                "a": a,
                "x": x,
                "prefactor": case(
                    "prefactor",
                    mx**ma * mp.e ** (-mx) / mp.gamma(ma),
                    float(stats.gamma.pdf(x, a) * x),
                ),
            }

    return out


def main() -> int:
    document = {
        "generator": "mpmath 50 dps for the certified value, scipy.special for the attainable accuracy",
        "regenerate": "python3 tools/generate_special_function_fixtures.py",
        "note": (
            "'certified' is the true value to 30 digits, from mpmath. 'attainable' is the "
            "log relative error SciPy reaches against it, in digits; null means SciPy hit it "
            "exactly. Vegoia is required to come within half a digit of SciPy."
        ),
        "log_gamma": log_gamma(),
        "error_function": error_function(),
        "incomplete_gamma": incomplete_gamma(),
        "incomplete_beta": incomplete_beta(),
        "gamma_prefactor": gamma_prefactor(),
    }

    path = pathlib.Path("resources/fixtures/stats/special_functions.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")

    counts = {k: len(v) for k, v in document.items() if isinstance(v, dict)}
    worst = []

    def scan(node, path_parts):
        if isinstance(node, dict):
            if "attainable" in node:
                if node["attainable"] is not None:
                    worst.append((node["attainable"], ".".join(path_parts)))
                return
            for k, v in node.items():
                scan(v, path_parts + [str(k)])

    scan(document, [])
    worst.sort()

    print(f"-> {path}")
    print(f"   {counts}")
    print(f"   {len(worst)} points with a measured ceiling")
    print("   SciPy's least accurate, in digits:")

    for value, where in worst[:8]:
        print(f"     {value:8.2f}  {where}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
