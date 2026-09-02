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
import networkx as nx
import numpy as np
from scipy import special, stats as sps

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


# Above these networkx takes minutes per operation and says nothing new; the
# ratio is already established on the graphs below them. The second limit is
# lower because betweenness and closeness are O(nm): on 5000 nodes and 28k
# edges that is 1.4e8 steps of interpreted Python, which runs for minutes.
NETWORKX_LIMIT = 5000
NETWORKX_QUADRATIC_LIMIT = 1000


def statistics_lane() -> dict:
    """The statistics operations, on inputs identical to the PHP side.

    Both sides build their inputs from the same closed forms in IEEE doubles,
    so the two are comparing the same arithmetic on the same numbers.

    What this does *not* control for, and should not: numpy and SciPy work on
    a whole array inside compiled code while Vegoia walks it in PHP. That is
    the comparison a caller actually faces -- it is why the sidecar exists --
    rather than an unfairness to correct for.
    """
    n = 1_000_000
    i = np.arange(n, dtype=np.float64)
    values = np.sin(i) * 1000.0 + 10_000_000.0
    paired = values * 0.5 + np.cos(i) * 250.0

    rows = np.arange(20_000, dtype=np.float64)
    columns = [np.sin(rows * (0.0007 * (j + 1) + 0.013)) for j in range(8)]
    design = np.column_stack([np.ones(20_000)] + columns)
    response = 3.0 + sum((j + 1) * 0.1 * c for j, c in enumerate(columns)) + np.cos(rows) * 0.01

    probabilities = np.arange(1, 100_001, dtype=np.float64) / 100_001.0

    def ols():
        q, r = np.linalg.qr(design)
        return np.linalg.solve(r, q.T @ response)

    return {
        # ddof=1: the sample standard deviation, as Descriptive reports.
        "stddev_1m": measure(lambda: np.std(values, ddof=1), runs=3),
        "pearson_1m": measure(lambda: np.corrcoef(values, paired)[0, 1], runs=3),
        # Householder QR, the decomposition Vegoia uses, rather than lstsq.
        "ols_20000x8": measure(ols, runs=3),
        "normal_quantile_100k": measure(lambda: sps.norm.ppf(probabilities), runs=3),
        "erfc_100k": measure(lambda: special.erfc(probabilities * 6.0), runs=3),
    }


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

        # networkx is the other comparison worth making. igraph answers what
        # compiled C costs; networkx answers what a library written in the
        # host language costs, which is the position Vegoia is actually in.
        # Its community detection is Louvain, not Leiden -- there is no Leiden
        # in networkx -- so that row compares the nearest available algorithm
        # and is labelled to say so.
        if nodes <= NETWORKX_LIMIT:
            nxg = nx.Graph()
            nxg.add_nodes_from(range(nodes))
            nxg.add_edges_from(edges)

            rows["nx_build"] = measure(lambda: nx.Graph(edges), runs=3, warmup=1)
            rows["nx_louvain"] = measure(
                lambda: nx.community.louvain_communities(nxg, seed=42), runs=1, warmup=0)
            rows["nx_pagerank"] = measure(lambda: nx.pagerank(nxg, alpha=0.85), runs=3, warmup=1)
            rows["nx_dijkstra"] = measure(
                lambda: nx.single_source_shortest_path_length(nxg, 0), runs=3, warmup=1)

        payload = json.dumps({"nodes": nodes,
                              "edges": [[u, v, 1.0] for u, v in edges]}).encode()

        def sidecar():
            subprocess.run([sys.executable, str(script)], input=payload,
                           stdout=subprocess.PIPE, check=True)

        rows["leiden_sidecar"] = measure(sidecar, runs=3, warmup=1)

        if nodes <= 5000:
            rows["betweenness"] = measure(lambda: g.betweenness(), runs=1, warmup=0)
            rows["closeness"] = measure(lambda: g.closeness(), runs=1, warmup=0)

            if nodes <= NETWORKX_QUADRATIC_LIMIT:
                rows["nx_betweenness"] = measure(
                    lambda: nx.betweenness_centrality(nxg), runs=1, warmup=0)
                rows["nx_closeness"] = measure(
                    lambda: nx.closeness_centrality(nxg), runs=1, warmup=0)

        for name, ms in rows.items():
            report.append({"graph": spec["name"], "nodes": nodes,
                           "edges": len(edges), "operation": name,
                           "median_ms": round(ms, 3)})
            if not as_json:
                print(f"{spec['name']:8s} {nodes:8d} {len(edges):9d} {name:>16s} {ms:12.2f}")
        if not as_json:
            print()

    stats = statistics_lane()

    if as_json:
        print(json.dumps({"graph": report, "stats": {k: round(v, 3) for k, v in stats.items()}},
                         indent=1))
    else:
        print(f"{'operation':24s} {'median ms':>12s}")
        print("-" * 37)
        for name, ms in stats.items():
            print(f"{name:24s} {ms:12.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
