#!/usr/bin/env python3
"""Compare two trees of fixture files, numerically rather than byte for byte.

The fixtures are golden values produced by networkx, igraph, leidenalg and
numpy. Regenerating them has to reproduce them, or they are a snapshot of one
machine rather than a reference -- but "reproduce" cannot mean byte equality.

That was tried first and failed in CI, and the diff said why: every difference
was one or two units in the last place. Pearson 0.8162365060002426 against
...2424, a HITS score of 0.24999999999999994 against 0.25, a correlation of
exactly zero against -8.4e-18. Those are not different answers; they are the
same answer summed in a different order. numpy dispatches its reduction loops
on the CPU's SIMD width, and the width sets the block size of the pairwise
summation, so the last bit moves with the hardware. numpy does not promise
bit-identical results across machines, and asking for it means the check can
only ever pass on the machine that produced the fixtures. (The exact
dispatch could not be confirmed locally: this CPU reports AVX512F=False, so
the wider path is not available here to compare against.)

So structure is compared exactly and numbers are compared with a tolerance,
which is what the fixtures actually claim. Anything non-numeric -- a
community membership, a triangle count, a key, an array length -- must match
exactly, because those are the things that move when a reference library
genuinely changes its mind.

One class of value is exempt and reported instead. Some fixtures record what
leidenalg's *search* found: the min/max/mean/stdev of modularity over fifty
seeded runs, and one partition taken at seed 42. Those are samples drawn from
another library's random number stream, and that stream is not part of its
API. Between leidenalg 0.10.2 and 0.12.0 every one of the 142 differences
seen here came from exactly those fields -- while the deterministic ones, all
1,100 of them, reproduced bit for bit, including the arithmetic of all five
quality functions at every resolution on fixed partitions. Requiring a
stochastic sample to reproduce would mean pinning leidenalg's RNG forever,
which tests nothing about this library.

The exemption is not a loophole, because the claim that matters is asserted
elsewhere and more strictly: the PHP suite requires this implementation to
reach the best modularity leidenalg reaches, on every fixture, and that bar
was verified to hold under both reference versions.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

# Golden values are asserted against by the test suite at tolerances no
# tighter than 1e-9. A thousandfold margin below that still leaves four
# orders of headroom over the ULP noise actually observed (~1e-16 relative).
RELATIVE = 1e-12
ABSOLUTE = 1e-12

# The attainable files are different in kind: they hold accuracy ceilings in
# digits, each one a measurement of somebody else's rounding error, so their
# own last digits are the least stable numbers in the whole tree. A one-ULP
# shift in numpy's answer moves a ceiling of 15.6 digits by a few tenths,
# because at that level the error being measured is itself one or two ULP.
# Half a digit is a factor of three in the error -- far above that noise, far
# below the whole digits a real change in accuracy costs.
DIGITS_TOLERANCE = 0.5

# Past this, the measurement is at the noise floor of double precision and
# says only "exact to the last bit". Two such measurements agree by being
# saturated, not by being equal.
DIGITS_FLOOR = 14.0


def is_digits_file(path: pathlib.Path) -> bool:
    return path.name.startswith("attainable_")


def is_sampled(where: str) -> bool:
    """Whether this value came out of leidenalg's stochastic search.

    `.leiden.` covers the per-objective bands; leiden_seed42 is the single
    partition pinned so the quality functions have a non-trivial input, and
    any partition serves that purpose. `seeds` is excluded: it is the number
    of runs the generator asked for, a constant, and it must not drift.
    """
    if where.endswith(".seeds"):
        return False

    return ".leiden." in where or ".quality_probes.leiden_seed42." in where


def compare(baseline, current, where: str, digits: bool, problems: list[str], sampled: list[str] | None = None) -> None:
    if type(baseline) is not type(current) and not (
        isinstance(baseline, (int, float))
        and isinstance(current, (int, float))
        and not isinstance(baseline, bool)
        and not isinstance(current, bool)
    ):
        problems.append(f"{where}: type changed, {type(baseline).__name__} -> {type(current).__name__}")
        return

    if sampled is not None and not isinstance(baseline, (dict, list)) and is_sampled(where):
        if baseline != current:
            sampled.append(f"{where}: {baseline!r} -> {current!r}")
        return

    if isinstance(baseline, dict):
        missing = sorted(set(baseline) - set(current))
        extra = sorted(set(current) - set(baseline))

        if missing:
            problems.append(f"{where}: keys disappeared: {', '.join(map(str, missing))}")
        if extra:
            problems.append(f"{where}: keys appeared: {', '.join(map(str, extra))}")

        for key in baseline:
            if key in current:
                compare(baseline[key], current[key], f"{where}.{key}", digits, problems, sampled)
        return

    if isinstance(baseline, list):
        if len(baseline) != len(current):
            problems.append(f"{where}: length {len(baseline)} -> {len(current)}")
            return

        for i, (b, c) in enumerate(zip(baseline, current)):
            compare(b, c, f"{where}[{i}]", digits, problems, sampled)
        return

    # Integers carry counts, community labels and node ids. They are exact
    # quantities and a change in one is a change in the answer.
    if isinstance(baseline, bool) or isinstance(baseline, str) or baseline is None or isinstance(baseline, int):
        if baseline != current:
            problems.append(f"{where}: {baseline!r} -> {current!r}")
        return

    if math.isnan(baseline) and math.isnan(current):
        return

    if digits:
        if baseline >= DIGITS_FLOOR and current >= DIGITS_FLOOR:
            return
        if abs(baseline - current) > DIGITS_TOLERANCE:
            problems.append(
                f"{where}: accuracy ceiling moved {baseline:.4f} -> {current:.4f} digits "
                f"({abs(baseline - current):.4f} > {DIGITS_TOLERANCE})"
            )
        return

    difference = abs(baseline - current)

    if difference <= ABSOLUTE:
        return
    if difference <= RELATIVE * max(abs(baseline), abs(current)):
        return

    problems.append(f"{where}: {baseline!r} -> {current!r} (differs by {difference:.3e})")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: compare_fixtures.py <baseline-dir> <current-dir>", file=sys.stderr)
        return 2

    baseline_root = pathlib.Path(sys.argv[1])
    current_root = pathlib.Path(sys.argv[2])

    baseline_files = {p.relative_to(baseline_root) for p in baseline_root.rglob("*.json")}
    current_files = {p.relative_to(current_root) for p in current_root.rglob("*.json")}

    problems: list[str] = []
    sampled: list[str] = []

    for missing in sorted(baseline_files - current_files):
        problems.append(f"{missing}: was not regenerated")

    for extra in sorted(current_files - baseline_files):
        problems.append(f"{extra}: appeared, and is not committed")

    checked = 0

    for name in sorted(baseline_files & current_files):
        with (baseline_root / name).open() as handle:
            baseline = json.load(handle)
        with (current_root / name).open() as handle:
            current = json.load(handle)

        compare(baseline, current, str(name), is_digits_file(name), problems, sampled)
        checked += 1

    if sampled:
        print(
            f"{len(sampled)} sampled value(s) drifted -- leidenalg explored in a different "
            "random order. Not a failure; see the module docstring. For example:"
        )

        for drift in sampled[:5]:
            print(f"  {drift}")

        print()

    if problems:
        print(f"Regenerating the fixtures changed them, in {len(problems)} place(s):\n", file=sys.stderr)

        for problem in problems[:40]:
            print(f"  {problem}", file=sys.stderr)

        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more", file=sys.stderr)

        return 1

    print(f"{checked} fixture files reproduce: structure identical, deterministic numbers within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
