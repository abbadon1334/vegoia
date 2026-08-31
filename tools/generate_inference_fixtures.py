#!/usr/bin/env python3
"""
Reference values for inference: p-values, and confidence and prediction intervals.

The library already computed the statistics; what was missing was the sentence
that follows one. An F of 5.12 on (3, 20) degrees of freedom is not a result
until someone says how often that happens by chance, and a coefficient of
1.03 is not a finding until someone says how wide the plausible range around
it is.

statsmodels is the reference here rather than SciPy alone, and it is a fair
one: it is what a Python user would reach for to answer these questions, it
computes the intervals from its own covariance matrix rather than from a
formula written out for the occasion, and it is independent of everything
else this library is checked against.

The inputs are the NIST StRD datasets already in the repository, so the
regressions being tested are the ones whose coefficients and standard errors
are certified to fifteen digits by an outside body. That matters: an interval
is the coefficient plus a multiple of its standard error, so a reference
interval computed from a bad fit would be a bad reference no matter how good
the arithmetic around it. Filip is included, the degree-ten polynomial that
several statistical packages have historically failed outright.

Usage: python3 tools/generate_inference_fixtures.py
"""

from __future__ import annotations

import json
import pathlib

import importlib.util
import warnings

import mpmath as mp
import numpy as np
import statsmodels.api as sm
from scipy import stats

mp.mp.dps = 50

ROOT = pathlib.Path(__file__).resolve().parent.parent
LLS = ROOT / "resources/fixtures/nist/lls"
ANOVA = ROOT / "resources/fixtures/nist/anova"

LEVELS = [0.90, 0.95, 0.99]


def lre_against(computed: float, exact) -> float | None:
    """Digits of agreement, or None when the two are equal."""
    if mp.mpf(computed) == exact:
        return None

    denominator = abs(exact) if exact != 0 else mp.mpf(1)

    return float(-mp.log10(abs(mp.mpf(computed) - exact) / denominator))


# The .dat reader lives in generate_lls_attainable.py and has already been
# taught the layout's surprises -- Filip's certified block being the only
# place its eleven parameters are written out, the polynomial datasets giving
# one column where the model wants its powers. Importing it rather than
# writing a second reader means the two tools cannot drift apart on what a
# dataset says.
_lls = importlib.util.spec_from_file_location("lls_attainable", ROOT / "tools/generate_lls_attainable.py")
lls_attainable = importlib.util.module_from_spec(_lls)
_lls.loader.exec_module(lls_attainable)


# Same reasoning for the analysis-of-variance layout, whose reader already
# knows that AtmWtAg.dat's header understates its own certified range.
_anova = importlib.util.spec_from_file_location(
    "anova_attainable", ROOT / "tools/generate_anova_attainable.py"
)
anova_attainable = importlib.util.module_from_spec(_anova)
_anova.loader.exec_module(anova_attainable)


def regression_section() -> dict:
    """Inference for each NIST regression, from the certified fit outwards.

    The reference t statistics, p-values and confidence intervals are computed
    from NIST's certified coefficients and standard errors, not from
    statsmodels' fit of the same data. That is deliberate and it matters for
    one dataset in particular.

    Filip is a degree-ten polynomial whose design matrix has numerical rank 10
    out of 11 in double precision -- which is exactly what it was published to
    demonstrate. statsmodels fits it through a pseudo-inverse and warns that
    the parameters are not uniquely determined, so its standard errors there
    are an artefact of that fallback rather than a reference anybody should be
    held to. Deriving the reference from the certified values instead gives a
    fixture that is as certified as NIST's own numbers, on every dataset,
    including the one designed to break things.

    statsmodels is still recorded alongside, as a second opinion on the
    datasets where its fit is sound: it confirms that the convention being
    used here -- two-sided p-values, intervals from the t distribution on the
    residual degrees of freedom -- is the one a statistician would expect.
    """
    out = {}

    for path in sorted(LLS.glob("*.dat")):
        spec = lls_attainable.parse(path)
        design, y = lls_attainable.design(spec)

        estimates = np.array(spec["estimates"], dtype=np.float64)
        errors = np.array(spec["errors"], dtype=np.float64)
        residual_df = len(y) - spec["n_params"]

        # Certified coefficient over certified standard error. Wampler1 and
        # Wampler2 are exact polynomial fits -- every residual is zero, so
        # every certified standard error is zero too, and the t statistic is
        # properly infinite with a p-value of zero. JSON has no infinity, so
        # those are recorded as null and the PHP side asserts INF and 0.0
        # rather than skipping them: dividing by a zero standard error is a
        # real case and it should not return NAN.
        t_statistics = [float(e / s) if s != 0 else None
                        for e, s in zip(estimates, errors)]
        p_values = [float(2 * stats.t.sf(abs(t), residual_df)) if t is not None else None
                    for t in t_statistics]

        entry = {
            "certified_coefficients": [float(v) for v in estimates],
            "certified_standard_errors": [float(v) for v in errors],
            "degrees_of_freedom": int(residual_df),
            "observations": int(len(y)),
            "parameters": int(spec["n_params"]),
            "has_intercept": bool(spec["intercept"]),
            "residual_standard_deviation": spec["resid_sd"],
            "t_statistics": t_statistics,
            "p_values": p_values,
            "confidence_intervals": {
                repr(level): [
                    [float(e - stats.t.isf((1 - level) / 2, residual_df) * s),
                     float(e + stats.t.isf((1 - level) / 2, residual_df) * s)]
                    for e, s in zip(estimates, errors)
                ]
                for level in LEVELS
            },
        }

        rank = int(np.linalg.matrix_rank(design))
        entry["rank_deficient"] = rank < design.shape[1]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.OLS(y, design).fit()

        # Nothing from statsmodels is recorded for a rank-deficient design.
        # Its fit there goes through a pseudo-inverse, and how the SVD treats a
        # singular value that is effectively zero depends on the LAPACK build,
        # so the numbers move between machines -- 309 of Filip's changed
        # between two CI runners on identical software. They also go unread:
        # the test skips those datasets precisely because a pseudo-inverse
        # describes the fallback rather than the model. Writing them down would
        # be storing irreproducible noise for nobody.
        if entry["rank_deficient"]:
            out[path.stem] = entry

            continue

        entry["statsmodels"] = {
            "t_statistics": [float(v) for v in fit.tvalues],
            "p_values": [float(v) for v in fit.pvalues],
            "confidence_intervals": {
                repr(level): [[float(low), float(high)]
                              for low, high in fit.conf_int(alpha=1 - level)]
                for level in LEVELS
            },
        }

        if spec["intercept"] and design.shape[1] > 1:
            entry["statsmodels"]["f_statistic"] = float(fit.fvalue)
            entry["statsmodels"]["f_p_value"] = float(fit.f_pvalue)

        # Predictions at three design points, split into the two things an
        # interval is made of, because they are answerable to different
        # references and mixing them hides both.
        #
        # The fitted value has a certified answer: it is the certified
        # coefficients against the design row, and mpmath evaluates that dot
        # product exactly. So it gets the same treatment as everything else
        # here -- a certified value, and statsmodels' distance from it as the
        # accuracy to reach. That matters on these datasets and not
        # academically: at the first row of Wampler5 the true fitted value is
        # exactly 1, statsmodels returns 1.0000011, and a test that compared
        # against statsmodels would have failed an implementation for being
        # nearly three digits closer to the truth.
        #
        # The half-width has no certified answer -- it needs the coefficient
        # covariance, which NIST does not certify -- so it is compared against
        # statsmodels directly, and lives under its key to say so.
        rows = [0, len(y) // 2, len(y) - 1]
        summary = fit.get_prediction(design[rows]).summary_frame(alpha=0.05)
        predictions = []

        for i, row in enumerate(rows):
            design_row = [float(v) for v in design[row]]
            exact = sum(mp.mpf(repr(c)) * mp.mpf(repr(d))
                        for c, d in zip(spec["estimates"], design_row))
            theirs = float(summary["mean"].iloc[i])

            mean_low = float(summary["mean_ci_lower"].iloc[i])
            mean_high = float(summary["mean_ci_upper"].iloc[i])
            obs_low = float(summary["obs_ci_lower"].iloc[i])
            obs_high = float(summary["obs_ci_upper"].iloc[i])

            predictions.append({
                "row": int(row),
                "design": design_row,
                "certified_fitted": mp.nstr(exact, 30, strip_zeros=False),
                "statsmodels_fitted": theirs,
                "statsmodels_fitted_accuracy": lre_against(theirs, exact),
                "statsmodels_mean_half_width": (mean_high - mean_low) / 2,
                "statsmodels_prediction_half_width": (obs_high - obs_low) / 2,
            })

        entry["statsmodels"]["predictions"] = predictions

        out[path.stem] = entry

    return out


def anova_section() -> dict:
    out = {}

    for path in sorted(ANOVA.glob("*.dat")):
        groups = anova_attainable.parse(path)["groups"]
        statistic, p = stats.f_oneway(*groups)

        out[path.stem] = {
            "f_statistic": float(statistic),
            "p_value": float(p),
            "between_degrees_of_freedom": len(groups) - 1,
            "within_degrees_of_freedom": sum(len(g) for g in groups) - len(groups),
        }

    return out


def main() -> int:
    document = {
        "generator": "statsmodels OLS on the NIST StRD datasets; scipy.stats.f_oneway for the ANOVA p-values",
        "regenerate": "python3 tools/generate_inference_fixtures.py",
        "note": (
            "Confidence intervals are two-sided at the stated level. Prediction intervals are for "
            "a single new observation; mean intervals are for the fitted mean at the same design "
            "point. Both are quoted at 95%."
        ),
        "levels": LEVELS,
        "regression": regression_section(),
        "anova": anova_section(),
    }

    path = ROOT / "resources/fixtures/stats/inference.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")

    print(f"-> {path}")
    print(f"   {len(document['regression'])} regressions, {len(document['anova'])} analyses of variance")

    for name, entry in document["regression"].items():
        finite = [p for p in entry["p_values"] if p is not None]
        flag = "  <- rank deficient, statsmodels values are a pseudo-inverse" \
            if entry["rank_deficient"] else ""
        smallest = f"{min(finite):.3g}" if finite else "exact fit, every p is zero"
        print(f"     {name:10s} {entry['degrees_of_freedom']:5d} df, "
              f"smallest p = {smallest}{flag}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
