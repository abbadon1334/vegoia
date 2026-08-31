#!/usr/bin/env python3
"""
Reference values for the four distributions that turn a statistic into a p-value.

Normal, Student's t, chi-squared and F. Between them they cover what the rest
of this library produces: a t for a regression coefficient, an F for an
analysis of variance or a nested model comparison, a chi-squared for a
likelihood ratio, a normal for anything asymptotic.

Certified values come from mpmath at 50 digits, built from the integrals
rather than from any library's answer:

    normal cdf  = erfc(-x/sqrt(2)) / 2
    chi2  cdf   = P(k/2, x/2)
    t     sf    = I_{k/(k+t^2)}(k/2, 1/2) / 2   for t > 0
    F     cdf   = I_{d1 f/(d1 f + d2)}(d1/2, d2/2)

SciPy is then measured against those, and what it reaches is recorded as the
attainable accuracy, exactly as the NIST datasets are handled elsewhere here.

Both tails are pinned separately, and that is the point of the file rather
than a detail of it. A p-value is a tail probability, and the tail is where
`1 - cdf` has no digits left: at t = 40 with 5 degrees of freedom the survival
function is about 6e-8, and at z = 10 it is 7.6e-24, which a subtraction from
one cannot represent at all. Quantiles are pinned in the far tail for the same
reason -- an inverse that goes through the cdf cannot return z = -10.

Usage: python3 tools/generate_distribution_fixtures.py
"""

from __future__ import annotations

import json
import pathlib

import mpmath as mp
from scipy import stats

mp.mp.dps = 50

DIGITS = 30
UNDERFLOW = mp.mpf("1e-300")


def certified(value: mp.mpf) -> str:
    return mp.nstr(value, DIGITS, strip_zeros=False)


def case(exact: mp.mpf, computed: float) -> dict:
    if exact != 0 and abs(exact) < UNDERFLOW:
        return {"certified": certified(exact), "scipy": float(computed),
                "attainable": None, "vanishes": True}

    if float(computed) == exact:
        attainable = None
    else:
        attainable = float(-mp.log10(abs((mp.mpf(computed) - exact) / exact))) if exact != 0 \
            else float(-mp.log10(abs(mp.mpf(computed)))) if computed != 0 else None

    return {"certified": certified(exact), "scipy": float(computed), "attainable": attainable}


# --- the four distributions, as exact functions of mpmath -------------------

def normal_cdf(x):
    return mp.erfc(-x / mp.sqrt(2)) / 2


def normal_sf(x):
    return mp.erfc(x / mp.sqrt(2)) / 2


def normal_pdf(x):
    return mp.exp(-x * x / 2) / mp.sqrt(2 * mp.pi)


def chi2_cdf(x, k):
    return mp.gammainc(k / 2, 0, x / 2, regularized=True)


def chi2_sf(x, k):
    return mp.gammainc(k / 2, x / 2, mp.inf, regularized=True)


def chi2_pdf(x, k):
    return mp.exp((k / 2 - 1) * mp.log(x) - x / 2 - mp.loggamma(k / 2) - (k / 2) * mp.log(2))


def t_sf(t, k):
    """Upper tail of Student's t, by the incomplete beta rather than a quadrature."""
    half = mp.betainc(k / 2, mp.mpf(1) / 2, 0, k / (k + t * t), regularized=True) / 2

    return half if t > 0 else 1 - half


def t_cdf(t, k):
    return 1 - t_sf(t, k) if t > 0 else t_sf(-t, k)


def t_pdf(t, k):
    return mp.exp(mp.loggamma((k + 1) / 2) - mp.loggamma(k / 2)
                  - mp.log(mp.sqrt(k * mp.pi)) - ((k + 1) / 2) * mp.log1p(t * t / k))


def f_cdf(x, d1, d2):
    return mp.betainc(d1 / 2, d2 / 2, 0, d1 * x / (d1 * x + d2), regularized=True)


def f_sf(x, d1, d2):
    return mp.betainc(d2 / 2, d1 / 2, 0, d2 / (d1 * x + d2), regularized=True)


def f_pdf(x, d1, d2):
    a, b = d1 / 2, d2 / 2
    z = d1 * x / (d1 * x + d2)

    return mp.exp(a * mp.log(z) + b * mp.log1p(-z) - mp.log(x)
                  - (mp.loggamma(a) + mp.loggamma(b) - mp.loggamma(a + b)))


def quantile(sf_exact, p, guess):
    """The x with survival(x) = p, found at 50 digits by bisection then Newton.

    Solved against the survival function, not the cdf: the quantiles worth
    pinning are the ones a p-value asks for, and those live where the cdf is
    1 to every digit a double has.
    """
    return mp.findroot(lambda x: sf_exact(x) - p, mp.mpf(guess))


# --- the argument grids -----------------------------------------------------

def normal_section() -> dict:
    points = [-10.0, -6.0, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.96, 2.5758, 3.0, 6.0, 10.0, 20.0]
    values = {}

    for x in points:
        m = mp.mpf(repr(x))
        values[repr(x)] = {
            "pdf": case(normal_pdf(m), float(stats.norm.pdf(x))),
            "cdf": case(normal_cdf(m), float(stats.norm.cdf(x))),
            "sf": case(normal_sf(m), float(stats.norm.sf(x))),
        }

    # Tail probabilities a test actually reports, down to where a double stops.
    quantiles = {}
    for p in [0.5, 0.25, 0.05, 0.025, 0.01, 0.001, 1e-6, 1e-12, 1e-30, 1e-100, 1e-300]:
        mp_p = mp.mpf(repr(p))
        exact = quantile(normal_sf, mp_p, float(stats.norm.isf(p)))
        quantiles[repr(p)] = case(exact, float(stats.norm.isf(p)))

    return {"points": values, "upper_quantiles": quantiles}


def student_t_section() -> dict:
    out = {}

    for k in [1.0, 2.0, 5.0, 10.0, 30.0, 100.0, 1000.0]:
        mk = mp.mpf(repr(k))
        points = {}

        for t in [-40.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 2.5, 5.0, 10.0, 40.0, 200.0]:
            mt = mp.mpf(repr(t))
            points[repr(t)] = {
                "pdf": case(t_pdf(mt, mk), float(stats.t.pdf(t, k))),
                "cdf": case(t_cdf(mt, mk), float(stats.t.cdf(t, k))),
                "sf": case(t_sf(mt, mk), float(stats.t.sf(t, k))),
            }

        quantiles = {}
        for p in [0.5, 0.05, 0.025, 0.005, 0.001, 1e-6, 1e-12]:
            mp_p = mp.mpf(repr(p))
            exact = quantile(lambda x, kk=mk: t_sf(x, kk), mp_p, float(stats.t.isf(p, k)))
            quantiles[repr(p)] = case(exact, float(stats.t.isf(p, k)))

        out[repr(k)] = {"points": points, "upper_quantiles": quantiles}

    return out


def chi_squared_section() -> dict:
    out = {}

    for k in [1.0, 2.0, 3.0, 10.0, 50.0, 500.0]:
        mk = mp.mpf(repr(k))
        points = {}

        for ratio in [0.01, 0.1, 0.5, 1.0, 1.5, 3.0, 10.0]:
            x = k * ratio
            mx = mp.mpf(repr(x))
            points[repr(x)] = {
                "pdf": case(chi2_pdf(mx, mk), float(stats.chi2.pdf(x, k))),
                "cdf": case(chi2_cdf(mx, mk), float(stats.chi2.cdf(x, k))),
                "sf": case(chi2_sf(mx, mk), float(stats.chi2.sf(x, k))),
            }

        quantiles = {}
        for p in [0.5, 0.05, 0.01, 0.001, 1e-6, 1e-12]:
            mp_p = mp.mpf(repr(p))
            exact = quantile(lambda x, kk=mk: chi2_sf(x, kk), mp_p, float(stats.chi2.isf(p, k)))
            quantiles[repr(p)] = case(exact, float(stats.chi2.isf(p, k)))

        out[repr(k)] = {"points": points, "upper_quantiles": quantiles}

    return out


def fisher_section() -> dict:
    out = {}

    for d1, d2 in [(1.0, 1.0), (2.0, 5.0), (3.0, 10.0), (5.0, 5.0), (10.0, 100.0), (100.0, 10.0)]:
        m1, m2 = mp.mpf(repr(d1)), mp.mpf(repr(d2))
        points = {}

        for x in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0, 1000.0]:
            mx = mp.mpf(repr(x))
            points[repr(x)] = {
                "pdf": case(f_pdf(mx, m1, m2), float(stats.f.pdf(x, d1, d2))),
                "cdf": case(f_cdf(mx, m1, m2), float(stats.f.cdf(x, d1, d2))),
                "sf": case(f_sf(mx, m1, m2), float(stats.f.sf(x, d1, d2))),
            }

        quantiles = {}
        for p in [0.5, 0.1, 0.05, 0.01, 0.001, 1e-6]:
            mp_p = mp.mpf(repr(p))
            exact = quantile(lambda x, a=m1, b=m2: f_sf(x, a, b), mp_p, float(stats.f.isf(p, d1, d2)))
            quantiles[repr(p)] = case(exact, float(stats.f.isf(p, d1, d2)))

        out[f"{d1!r},{d2!r}"] = {"points": points, "upper_quantiles": quantiles}

    return out


def main() -> int:
    document = {
        "generator": "mpmath 50 dps for the certified value, scipy.stats for the attainable accuracy",
        "regenerate": "python3 tools/generate_distribution_fixtures.py",
        "note": (
            "'certified' is the true value to 30 digits. 'attainable' is the log relative error "
            "SciPy reaches against it; null means SciPy hit it exactly. Both tails are pinned "
            "separately because a p-value lives in the one that 1 - cdf cannot represent."
        ),
        "normal": normal_section(),
        "student_t": student_t_section(),
        "chi_squared": chi_squared_section(),
        "fisher": fisher_section(),
    }

    path = pathlib.Path("resources/fixtures/stats/distributions.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")

    worst = []

    def scan(node, parts):
        if isinstance(node, dict):
            if "certified" in node:
                if node["attainable"] is not None:
                    worst.append((node["attainable"], ".".join(parts)))
                return
            for k, v in node.items():
                scan(v, parts + [str(k)])

    scan(document, [])
    worst.sort()

    print(f"-> {path}")
    print(f"   {len(worst)} points with a measured ceiling")
    print("   SciPy's least accurate, in digits:")

    for value, where in worst[:8]:
        print(f"     {value:8.2f}  {where}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
