#!/usr/bin/env python3
"""
Measure the accuracy ceiling for linear least squares on the NIST StRD sets.

The NIST regression datasets are deliberately ill-conditioned. Filip is a
degree-10 polynomial whose design matrix has a condition number around 1e15 --
NIST notes that many statistical packages fail to fit it at all, and the normal
equations, which square the condition number, cannot touch it.

Vegoia solves via Householder QR. To keep the comparison honest this script
measures numpy solving the *same* way, rather than with SVD, so any gap is a
gap in the implementation and not in the choice of algorithm.

Usage:  python3 tools/generate_lls_attainable.py
"""

from __future__ import annotations

import json
import math
import pathlib
import re

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
LLS = ROOT / "resources/fixtures/nist/lls"


def parse(path: pathlib.Path) -> dict:
    lines = path.read_text().splitlines()
    head = "\n".join(lines[:20])

    cf, ct = map(int, re.search(r"Certified Values\s*\(lines\s+(\d+)\s+to\s+(\d+)\)", head).groups())
    df, dt = map(int, re.search(r"Data\s*\(lines\s+(\d+)\s+to\s+(\d+)\)", head).groups())

    model = next(l for l in lines if re.match(r"\s+y\s*=", l))
    n_pred = int(re.search(r"(\d+)\s+Predictor Variable", head).group(1))

    # The parameter count comes from the certified block, not the model line:
    # Filip writes "B0 + B1*x + B2*(x**2) + ... + B9*(x**9) + B10*(x**10)", and
    # those literal dots hide B3 through B8. The certified rows are explicit.
    indices, estimates, errors = [], [], []
    for line in lines[cf - 1:ct]:
        m = re.match(r"\s*B(\d+)\s+(-?[\d.]+(?:[Ee][-+]?\d+)?)\s+(-?[\d.]+(?:[Ee][-+]?\d+)?)", line)
        if m:
            indices.append(int(m.group(1)))
            estimates.append(float(m.group(2)))
            errors.append(float(m.group(3)))

    intercept = 0 in indices
    n_params = len(indices)

    resid_sd, r2 = None, None
    for line in lines[cf - 1:ct]:
        s = line.strip()
        if s.startswith("Standard Deviation") and resid_sd is None:
            resid_sd = float(re.search(r"(-?[\d.]+(?:[Ee][-+]?\d+)?)\s*$", s).group(1))
        if s.startswith("R-Squared"):
            r2 = float(re.search(r"(-?[\d.]+(?:[Ee][-+]?\d+)?)\s*$", s).group(1))

    rows = [[float(p) for p in l.split()] for l in lines[df - 1:dt] if l.split()]
    return dict(model=model.strip(), intercept=intercept, n_params=n_params,
                n_pred=n_pred, estimates=estimates, errors=errors,
                resid_sd=resid_sd, r2=r2, data=np.array(rows, dtype=np.float64))


def design(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    data = spec["data"]
    y, x = data[:, 0], data[:, 1:]

    # One predictor but more than two parameters means the model is polynomial
    # in that predictor, not linear in several.
    if spec["n_pred"] == 1 and spec["n_params"] > 2:
        degree = spec["n_params"] - 1
        X = np.column_stack([x[:, 0] ** k for k in range(1, degree + 1)])
    else:
        X = x

    if spec["intercept"]:
        X = np.column_stack([np.ones(len(y)), X])
    return X, y


def lre(computed: float, certified: float) -> float | None:
    if computed == certified:
        return None
    d = abs(certified) if certified != 0 else 1.0
    return -math.log10(abs(computed - certified) / d)


def worst(pairs) -> float | None:
    scored = [v for v in (lre(float(a), b) for a, b in pairs) if v is not None]
    return min(scored) if scored else None


def main() -> int:
    out = {}
    for path in sorted(LLS.glob("*.dat")):
        spec = parse(path)
        X, y = design(spec)

        Q, R = np.linalg.qr(X)                  # Householder QR, as Vegoia does
        beta = np.linalg.solve(R, Q.T @ y)

        residuals = y - X @ beta
        dof = len(y) - X.shape[1]
        sigma2 = float(residuals @ residuals) / dof
        # cov = sigma^2 (R^-1)(R^-1)^T. Forming R^T R first would square the
        # condition number -- exactly the mistake QR was chosen to avoid -- and
        # on Filip it yields negative variances and a sqrt of NaN.
        R_inv = np.linalg.inv(R)
        se = math.sqrt(sigma2) * np.sqrt((R_inv ** 2).sum(axis=1))

        out[path.stem] = {
            "coefficients": worst(zip(beta, spec["estimates"])),
            "standardErrors": worst(zip(se, spec["errors"])),
            "residualStandardDeviation": lre(math.sqrt(sigma2), spec["resid_sd"]),
        }
        fmt = lambda z: "exact" if z is None else f"{z:6.2f}"
        print(f"{path.stem:10s} p={X.shape[1]:2d} n={len(y):4d}  "
              f"beta={fmt(out[path.stem]['coefficients'])}  "
              f"se={fmt(out[path.stem]['standardErrors'])}  "
              f"sd={fmt(out[path.stem]['residualStandardDeviation'])}")

    doc = {
        "generator": f"numpy {np.__version__}, Householder QR (np.linalg.qr)",
        "meaning": (
            "Worst-case Log Relative Error across the parameters of each NIST "
            "linear least squares dataset, for an independent implementation "
            "using the same decomposition. Vegoia must match these."
        ),
        "regenerate": "python3 tools/generate_lls_attainable.py",
        "datasets": out,
    }
    target = LLS.parent / "attainable_lls.json"
    target.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"\n-> {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
