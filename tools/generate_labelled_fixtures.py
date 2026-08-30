#!/usr/bin/env python3
"""
Golden fixtures for graphs whose *correct answer* is known.

Everything else in this suite compares Vegoia against a reference
implementation. That proves agreement, not competence: two implementations can
agree on a mediocre partition. These fixtures instead carry a ground truth --
the communities the graph was built from, or the departments people actually
belong to -- so the question becomes "does it find the right answer", which is
the question that decides whether this library can replace the Python one.

Two sources:

  * LFR benchmark (Lancichinetti-Fortunato-Radicchi), the standard synthetic
    benchmark for community detection. The mixing parameter mu is the fraction
    of each node's edges that leave its community, so it dials difficulty
    directly: at 0.1 the structure is obvious, at 0.6 it is nearly gone. An
    algorithm that only works on easy graphs is exposed by the sweep.

  * email-Eu-core (SNAP), a real email network of 1005 people in 42 research
    departments. Real graphs are messier than any generator: departments
    overlap, some are tiny, and no algorithm recovers them perfectly -- which
    is the point. The recorded scores are what leidenalg achieves, so the bar
    is "as good as the reference", not "perfect".

Agreement is scored with NMI and ARI, the two measures the literature uses.
Both are computed by scikit-learn here, so the PHP implementations of them are
themselves tested against an independent oracle.

Usage: python3 tools/generate_labelled_fixtures.py [outdir]
"""

from __future__ import annotations

import gzip
import json
import pathlib
import statistics
import sys
import urllib.request

import igraph as ig
import leidenalg
from networkx.generators.community import LFR_benchmark_graph
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

SEEDS = list(range(1, 21))
SNAP = "https://snap.stanford.edu/data/"


def leiden_scores(g: ig.Graph, truth: list[int]) -> dict:
    """What the reference implementation recovers, across many seeds."""
    nmi, ari, ks, qs = [], [], [], []

    for seed in SEEDS:
        p = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=seed)
        nmi.append(normalized_mutual_info_score(truth, p.membership))
        ari.append(adjusted_rand_score(truth, p.membership))
        ks.append(len(set(p.membership)))
        qs.append(g.modularity(p.membership))

    return {
        "nmi": {"min": min(nmi), "max": max(nmi), "mean": statistics.fmean(nmi)},
        "ari": {"min": min(ari), "max": max(ari), "mean": statistics.fmean(ari)},
        "communities": {"min": min(ks), "max": max(ks)},
        "modularity": {"min": min(qs), "max": max(qs)},
        "seeds": len(SEEDS),
    }


def lfr(nodes: int, mu: float, seed: int) -> tuple[list[list[int]], list[int]]:
    G = LFR_benchmark_graph(
        n=nodes, tau1=3, tau2=1.5, mu=mu,
        average_degree=8, min_community=int(nodes / 12), seed=seed,
    )
    order = {v: i for i, v in enumerate(sorted(G.nodes()))}

    # Node attribute 'community' is the frozen set it belongs to; number them
    # by first appearance so the labels are stable across runs.
    label_of, truth = {}, [0] * len(order)
    for v in sorted(G.nodes()):
        key = frozenset(G.nodes[v]["community"])
        if key not in label_of:
            label_of[key] = len(label_of)
        truth[order[v]] = label_of[key]

    edges = sorted({tuple(sorted((order[u], order[v]))) for u, v in G.edges()})
    return [[u, v] for u, v in edges], truth


def email_eu_core(cache: pathlib.Path) -> tuple[list[list[int]], list[int]]:
    cache.mkdir(parents=True, exist_ok=True)
    files = {}

    for name in ("email-Eu-core.txt.gz", "email-Eu-core-department-labels.txt.gz"):
        path = cache / name
        if not path.exists():
            urllib.request.urlretrieve(SNAP + name, path)
        files[name] = gzip.decompress(path.read_bytes()).decode()

    truth_by_node = {}
    for line in files["email-Eu-core-department-labels.txt.gz"].split("\n"):
        if line.strip():
            node, dept = line.split()
            truth_by_node[int(node)] = int(dept)

    order = {v: i for i, v in enumerate(sorted(truth_by_node))}
    truth = [0] * len(order)
    for node, dept in truth_by_node.items():
        truth[order[node]] = dept

    pairs = set()
    for line in files["email-Eu-core.txt.gz"].split("\n"):
        if not line.strip():
            continue
        u, v = (int(x) for x in line.split())
        if u != v and u in order and v in order:      # drop self-loops
            pairs.add(tuple(sorted((order[u], order[v]))))

    return [[u, v] for u, v in sorted(pairs)], truth


def build(name: str, edges: list[list[int]], truth: list[int], note: str) -> dict:
    g = ig.Graph(n=len(truth), edges=[(e[0], e[1]) for e in edges])
    return {
        "name": name,
        "note": note,
        "nodes": len(truth),
        "edges": edges,
        "ground_truth": truth,
        "ground_truth_communities": len(set(truth)),
        "ground_truth_modularity": g.modularity(truth),
        "reference": leiden_scores(g, truth),
    }


def main() -> int:
    outdir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "resources/fixtures/labelled")
    outdir.mkdir(parents=True, exist_ok=True)
    cache = pathlib.Path("/tmp/vegoia-snap")

    index = []

    for mu in (0.1, 0.3, 0.5):
        name = f"lfr_400_mu{int(mu * 10):02d}"
        edges, truth = lfr(400, mu, seed=42)
        fx = build(name, edges, truth,
                   f"LFR benchmark, 400 nodes, mixing parameter {mu}. "
                   f"{'Obvious' if mu <= 0.2 else 'Moderate' if mu <= 0.4 else 'Hard'} community structure.")
        (outdir / f"{name}.json").write_text(json.dumps(fx, indent=1, sort_keys=True) + "\n")
        index.append({"name": name, "nodes": fx["nodes"], "edges": len(edges),
                      "communities": fx["ground_truth_communities"]})
        r = fx["reference"]
        print(f"{name:20s} n={fx['nodes']:5d} m={len(edges):6d} truth={fx['ground_truth_communities']:3d} "
              f"NMI=[{r['nmi']['min']:.3f},{r['nmi']['max']:.3f}] ARI=[{r['ari']['min']:.3f},{r['ari']['max']:.3f}]")

    edges, truth = email_eu_core(cache)
    fx = build("email_eu_core", edges, truth,
               "SNAP email-Eu-core: 1005 people, 42 real research departments. "
               "Ground truth is messy on purpose -- nobody recovers it exactly.")
    (outdir / "email_eu_core.json").write_text(json.dumps(fx, indent=1, sort_keys=True) + "\n")
    index.append({"name": "email_eu_core", "nodes": fx["nodes"], "edges": len(edges),
                  "communities": fx["ground_truth_communities"]})
    r = fx["reference"]
    print(f"{'email_eu_core':20s} n={fx['nodes']:5d} m={len(edges):6d} truth={fx['ground_truth_communities']:3d} "
          f"NMI=[{r['nmi']['min']:.3f},{r['nmi']['max']:.3f}] ARI=[{r['ari']['min']:.3f},{r['ari']['max']:.3f}]")

    (outdir / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    print(f"\n{len(index)} fixtures -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
