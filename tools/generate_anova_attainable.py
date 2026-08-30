#!/usr/bin/env python3
"""
Accuracy ceilings for the NIST ANOVA datasets, measured with SciPy.

The SmLs series exists to break ANOVA implementations: within each series the
datasets hold the same structure at increasing digit counts, so SmLs01 is
benign and SmLs09 is deliberately at the edge of what binary64 can carry. The
certified F statistic is 21 exactly for all of them; how many digits of it
survive is the measurement.

As elsewhere in this suite, the threshold is what an independent
implementation reaches on the same data rather than a constant chosen until
the tests passed.

Usage: python3 tools/generate_anova_attainable.py
"""

from __future__ import annotations

import json
import math
import pathlib
import re

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
ANOVA = ROOT / "resources/fixtures/nist/anova"


def parse(path: pathlib.Path) -> dict:
    lines = path.read_text().splitlines()
    head = "\n".join(lines[:20])

    cf, ct = map(int, re.search(r"Certified Values\s*\(lines\s+(\d+)\s+to\s+(\d+)\)", head).groups())
    df, dt = map(int, re.search(r"Data\s*\(lines\s+(\d+)\s+to\s+(\d+)\)", head).groups())

    groups: dict[str, list[float]] = {}
    for line in lines[df - 1:dt]:
        parts = line.split()
        if len(parts) >= 2:
            groups.setdefault(parts[0], []).append(float(parts[1]))

    certified = {}
    for line in lines[cf - 1:ct]:
        s = line.strip()
        m = re.match(r"^(Between|Within)\s+\S+\s+(\d+)\s+(\S+)\s+(\S+)(?:\s+(\S+))?", s)
        if m:
            certified[m.group(1).lower()] = {
                "df": int(m.group(2)), "ss": float(m.group(3)), "ms": float(m.group(4)),
                "f": float(m.group(5)) if m.group(5) else None,
            }
        if "R-Squared" in s:
            certified["r2"] = float(re.search(r"(\S+)\s*$", s).group(1))

    # Searched over the whole file rather than the declared certified range,
    # because AtmWtAg.dat declares lines 41-47 and puts its residual standard
    # deviation on line 48. The header is wrong in the source data; trusting it
    # loses the value on that one dataset.
    for line in lines:
        s = line.strip()
        if s.startswith("Standard Deviation"):
            certified["sd"] = float(re.search(r"(\S+)\s*$", s).group(1))
            break

    return {"groups": list(groups.values()), "certified": certified}


def lre(computed: float, certified: float) -> float | None:
    if computed == certified:
        return None
    d = abs(certified) if certified != 0 else 1.0
    return -math.log10(abs(computed - certified) / d)


def main() -> int:
    out = {}
    for path in sorted(ANOVA.glob("*.dat")):
        spec = parse(path)
        groups = [np.array(g, dtype=np.float64) for g in spec["groups"]]
        cert = spec["certified"]

        f = float(stats.f_oneway(*groups).statistic)

        allv = np.concatenate(groups)
        grand = allv.mean()
        between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        within = sum(float(((g - g.mean()) ** 2).sum()) for g in groups)
        k, n = len(groups), len(allv)
        sd = math.sqrt(within / (n - k))
        r2 = between / (between + within)

        out[path.stem] = {
            "fStatistic": lre(f, cert["between"]["f"]),
            "rSquared": lre(r2, cert["r2"]),
            "residualStandardDeviation": lre(sd, cert["sd"]),
        }
        fmt = lambda z: "exact" if z is None else f"{z:6.2f}"
        print(f"{path.stem:10s} F={fmt(out[path.stem]['fStatistic'])}  "
              f"R2={fmt(out[path.stem]['rSquared'])}  sd={fmt(out[path.stem]['residualStandardDeviation'])}")

    doc = {
        "generator": f"scipy f_oneway with numpy {np.__version__}, IEEE-754 binary64",
        "meaning": (
            "Log Relative Error an independent float64 implementation attains on each "
            "certified ANOVA value. Vegoia is required to match these."
        ),
        "regenerate": "python3 tools/generate_anova_attainable.py",
        "datasets": out,
    }
    target = ANOVA.parent / "attainable_anova.json"
    target.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"\n-> {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
