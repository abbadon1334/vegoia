#!/usr/bin/env python3
"""
Time the Python reference stack on the same graphs, for comparison.

Two numbers are reported for community detection, and the difference between
them is the point:

  * `leiden`         -- igraph/leidenalg computing, and nothing else. This is
                        compiled C, and it is the number PHP has to live with.
  * `leiden_sidecar` -- the same work as an application would actually get it:
                        spawn `python3`, import igraph and leidenalg, hand the
                        graph over as JSON, parse the answer back. This is what
                        a PHP application shelling out to Python really pays,
                        per call.

Usage: python3 tools/bench_reference.py [graphdir] [--json]
"""

from __future__ import annotations

import json
import pathlib
import statistics
import subprocess
import sys
import tempfile
import time

import igraph as ig
import leidenalg

SIDECAR = r'''
import json, sys
import igraph as ig
import leidenalg
payload = json.load(sys.stdin)
g = ig.Graph(n=payload["nodes"], edges=[(e[0], e[1]) for e in payload["edges"]])
p = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=42)
json.dump({"membership": p.membership, "modularity": p.modularity}, sys.stdout)
'''


def measure(fn, runs=7, warmup=2):
    for _ in range(warmup):
        fn()
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(timings)


def load(path: str):
    nodes, edges = 0, []
    with open(path) as handle:
        for line in handle:
            if line.startswith("#"):
                nodes = int(line.split("\t")[1])
                continue
            u, v, _w = line.split("\t")
            edges.append((int(u), int(v)))
    return nodes, edges


def main() -> int:
    directory = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vegoia-bench"
    as_json = "--json" in sys.argv
    index = json.loads((pathlib.Path(directory) / "index.json").read_text())

    script = pathlib.Path(tempfile.gettempdir()) / "vegoia_sidecar.py"
    script.write_text(SIDECAR)

    report = []
    if not as_json:
        print(f"{'graph':8s} {'nodes':>8s} {'edges':>9s} {'operation':>16s} {'median ms':>12s}")
        print("-" * 58)

    for spec in index:
        nodes, edges = load(spec["file"])
        g = ig.Graph(n=nodes, edges=edges)

        rows = {
            "build": measure(lambda: ig.Graph(n=nodes, edges=edges)),
            "leiden": measure(lambda: leidenalg.find_partition(
                g, leidenalg.ModularityVertexPartition, seed=42)),
            "pagerank": measure(lambda: g.pagerank(damping=0.85)),
            "dijkstra": measure(lambda: g.distances(source=0)),
        }

        payload = json.dumps({"nodes": nodes,
                              "edges": [[u, v, 1.0] for u, v in edges]}).encode()

        def sidecar():
            subprocess.run([sys.executable, str(script)], input=payload,
                           stdout=subprocess.PIPE, check=True)

        rows["leiden_sidecar"] = measure(sidecar, runs=3, warmup=1)

        if nodes <= 5000:
            rows["betweenness"] = measure(lambda: g.betweenness(), runs=1, warmup=0)
            rows["closeness"] = measure(lambda: g.closeness(), runs=1, warmup=0)

        for name, ms in rows.items():
            report.append({"graph": spec["name"], "nodes": nodes,
                           "edges": len(edges), "operation": name,
                           "median_ms": round(ms, 3)})
            if not as_json:
                print(f"{spec['name']:8s} {nodes:8d} {len(edges):9d} {name:>16s} {ms:12.2f}")
        if not as_json:
            print()

    if as_json:
        print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
