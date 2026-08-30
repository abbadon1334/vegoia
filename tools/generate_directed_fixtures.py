#!/usr/bin/env python3
"""
Golden fixtures for directed graphs.

Directed graphs need their own file because most measures mean something
different on them, and one -- HITS -- only means anything on them at all: hub
and authority scores coincide on an undirected graph, and on a bipartite one
networkx's implementation goes numerically unstable and returns negative
scores. Testing HITS against those values would have pinned nonsense.

Usage: python3 tools/generate_directed_fixtures.py [outdir]
"""

from __future__ import annotations

import json
import pathlib
import sys

import networkx as nx
import numpy as np


def is_well_posed(M) -> bool:
    """Whether the leading eigenvalue is simple.

    HITS is the principal eigenvector, so it is only defined when that
    eigenvector is unique -- which requires the largest eigenvalue to be
    strictly larger than the next. On a directed cycle A A^T is the identity
    and every eigenvalue is 1; on a bow tie the largest has multiplicity three.
    In those cases every implementation returns something arbitrary, so
    recording a golden value would pin an accident of the solver.
    """
    values = sorted(np.linalg.eigvalsh(M), reverse=True)

    if len(values) < 2 or values[0] <= 1e-12:
        return False

    return (values[0] - values[1]) / values[0] > 1e-6


def principal_eigenvector(M) -> list[float]:
    """Non-negative principal eigenvector, normalised to sum 1.

    Perron-Frobenius guarantees the leading eigenvector of a non-negative
    matrix can be taken non-negative; eigh may return it negated, so the sign
    is fixed here.
    """
    values, vectors = np.linalg.eigh(M)
    v = vectors[:, int(np.argmax(values))]

    if v.sum() < 0:
        v = -v

    v = np.clip(v, 0.0, None)
    total = v.sum()

    return [float(x) for x in (v / total if total > 0 else np.full(len(v), 1.0 / len(v)))]


def build(name: str, edges: list[tuple[int, int]], nodes: int, note: str) -> dict:
    G = nx.DiGraph()
    G.add_nodes_from(range(nodes))
    G.add_edges_from(edges)

    def vec(d):
        return [float(d.get(v, 0.0)) for v in range(nodes)]

    # Computed as eigenvectors rather than via nx.hits(). At tol=1e-14 that
    # implementation goes numerically unstable and returns negative scores and
    # scores above 1, which HITS cannot produce -- it is a non-negative
    # probability-like vector by construction. The definition itself is
    # unambiguous: hubs are the principal eigenvector of A A^T, authorities of
    # A^T A, and eigh on a symmetric matrix is exact and stable.
    A = nx.to_numpy_array(G, nodelist=range(nodes))
    hub_matrix, authority_matrix = A @ A.T, A.T @ A
    well_posed = is_well_posed(hub_matrix) and is_well_posed(authority_matrix)
    hubs = principal_eigenvector(hub_matrix) if well_posed else None
    authorities = principal_eigenvector(authority_matrix) if well_posed else None

    return {
        "name": name,
        "note": note,
        "nodes": nodes,
        "directed": True,
        "edges": [[u, v, 1.0] for u, v in sorted(edges)],
        "expected": {
            "pagerank": vec(nx.pagerank(G, alpha=0.85, tol=1.0e-12, max_iter=1000)),
            "betweenness": vec(nx.betweenness_centrality(G, normalized=False)),
            "hits_hubs": hubs,
            "hits_authorities": authorities,
            "out_degree": vec(dict(G.out_degree())),
            "in_degree": vec(dict(G.in_degree())),
        },
    }


# HITS values are only recorded where the principal eigenvector is unique.
# On a directed cycle A A^T is the identity: every eigenvalue is 1, every
# vector is an eigenvector, and the scores are not defined at all -- any
# implementation returns something arbitrary there and pinning it would be
# pinning an accident. The same degeneracy appears on a plain chain. Those
# shapes are still exercised, by property rather than by golden value.
CATALOGUE = [
    ("hub_and_authority", [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (2, 4), (2, 5)], 6,
     "Kleinberg's shape: 0-2 only point, 3-5 only receive. The two scores must separate."),
    ("star_out", [(0, 1), (0, 2), (0, 3), (0, 4)], 5,
     "One pure hub, four pure authorities."),
    ("bow_tie", [(0, 2), (1, 2), (2, 3), (2, 4), (3, 5), (4, 5)], 6,
     "Two sources, a bridge, two sinks: hub and authority peak at different nodes."),
    ("chain", [(0, 1), (1, 2), (2, 3)], 4,
     "A directed path. HITS is degenerate here; PageRank and betweenness are not."),
    ("two_components", [(0, 1), (1, 0), (2, 3)], 4,
     "Disconnected and directed: PageRank's teleportation has to carry it."),
]




def main() -> int:
    outdir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "resources/fixtures/directed")
    outdir.mkdir(parents=True, exist_ok=True)

    index = []
    for name, edges, nodes, note in CATALOGUE:
        fx = build(name, edges, nodes, note)
        (outdir / f"{name}.json").write_text(json.dumps(fx, indent=1, sort_keys=True) + "\n")
        index.append({"name": name, "nodes": nodes, "edges": len(edges)})
        e = fx["expected"]
        posed = e["hits_hubs"] is not None
        print(f"{name:20s} n={nodes} m={len(edges):3d}  HITS: "
              + (f"hubs[0]={e['hits_hubs'][0]:.4f} auth[0]={e['hits_authorities'][0]:.4f}"
                 if posed else "degenerate, not recorded"))

    (outdir / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    print(f"\n{len(index)} fixtures -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
