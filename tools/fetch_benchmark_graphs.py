#!/usr/bin/env python3
"""
Fetch the graphs the literature actually benchmarks on.

The synthetic planted-partition graphs in bench_graphs.php are useful because
they are reproducible and their community structure is known, but they are
regular in a way real networks are not: degrees are near-uniform, there are no
hubs, and the clustering coefficient is whatever the model gives. Real graphs
are heavy-tailed, and heavy tails are what decide whether an implementation
holds up -- a single node with 10,000 neighbours exercises code paths that a
degree-11 graph never touches.

These are the SNAP collection's standard collaboration and social graphs, the
same files used in the igraph and networkx performance literature, so timings
here can be compared against published numbers rather than only against each
other.

SNAP files list each undirected edge twice and number nodes arbitrarily, so
both are normalised: deduplicated, and renumbered 0..n-1 in order of first
appearance.

Usage: python3 tools/fetch_benchmark_graphs.py [outdir]
"""

from __future__ import annotations

import gzip
import json
import pathlib
import sys
import urllib.request

BASE = "https://snap.stanford.edu/data/"

GRAPHS = [
    ("facebook_combined", "facebook_combined.txt.gz", "Social circles from Facebook. Small, dense, heavily clustered."),
    ("ca-GrQc", "ca-GrQc.txt.gz", "Collaboration network, general relativity. The standard small benchmark."),
    ("ca-HepTh", "ca-HepTh.txt.gz", "Collaboration network, high energy physics theory."),
    ("ca-CondMat", "ca-CondMat.txt.gz", "Collaboration network, condensed matter."),
    ("email-Enron", "email-Enron.txt.gz", "Enron email. Heavy-tailed with extreme hubs."),
    ("com-amazon", "bigdata/communities/com-amazon.ungraph.txt.gz", "Amazon co-purchasing, with ground-truth communities."),
    ("com-dblp", "bigdata/communities/com-dblp.ungraph.txt.gz", "DBLP co-authorship, with ground-truth communities."),
]


def fetch(path: str, cache: pathlib.Path) -> str:
    target = cache / path.rsplit("/", 1)[-1]

    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {path} ...", flush=True)
        urllib.request.urlretrieve(BASE + path, target)

    return gzip.decompress(target.read_bytes()).decode()


def main() -> int:
    outdir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/vegoia-bench-real")
    outdir.mkdir(parents=True, exist_ok=True)
    cache = pathlib.Path("/tmp/vegoia-snap-cache")

    index = []

    for name, path, note in GRAPHS:
        text = fetch(path, cache)

        renumber: dict[int, int] = {}
        edges = set()

        for line in text.split("\n"):
            if not line or line[0] == "#":
                continue
            a, b = line.split()[:2]
            u, v = int(a), int(b)
            if u == v:
                continue                       # self-loops carry no structure here
            for node in (u, v):
                if node not in renumber:
                    renumber[node] = len(renumber)
            lo, hi = sorted((renumber[u], renumber[v]))
            edges.add((lo, hi))                # SNAP lists each edge twice

        nodes = len(renumber)
        target = outdir / f"{name}.tsv"

        with target.open("w") as handle:
            handle.write(f"# nodes\t{nodes}\n")
            for u, v in sorted(edges):
                handle.write(f"{u}\t{v}\t1.0\n")

        index.append({"name": name, "nodes": nodes, "edges": len(edges),
                      "file": str(target), "note": note})
        print(f"{name:<20} n={nodes:7d} m={len(edges):8d}  {note}")

    (outdir / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    print(f"\n{len(index)} graphs -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
