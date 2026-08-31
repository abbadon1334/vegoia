#!/usr/bin/env python3
"""
Reference values for minimum and maximum spanning forests, from NetworkX.

A spanning tree keeps the cheapest set of edges that still joins everything.
Its use here is the opposite one: given a similarity graph -- k nearest
neighbours over embeddings, say -- the *maximum* spanning tree keeps the
strongest link out of every node and throws away the rest, which turns a dense
noisy graph into a skeleton you can look at.

What is pinned is the total weight and the edge count, not the edge set. The
tree is not unique when weights tie, and every fixture here is unweighted or
nearly so, which means ties everywhere: any implementation that broke them
differently would produce a different tree of exactly the same weight.
Pinning the edges would be pinning NetworkX's tie-breaking, which is not a
property of the answer.

Disconnected graphs give a forest rather than a tree, one tree per component,
and that is the general case rather than an error -- so the edge count is
n - c, not n - 1.

Usage: python3 tools/generate_spanning_tree_fixtures.py
"""

from __future__ import annotations

import json
import pathlib

import networkx as nx

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPHS = ROOT / "resources/fixtures/graph"


def load(name: str) -> nx.Graph:
    document = json.loads((GRAPHS / f"{name}.json").read_text())
    graph = nx.Graph()
    graph.add_nodes_from(range(document["nodes"]))

    for edge in document["edges"]:
        graph.add_edge(int(edge[0]), int(edge[1]),
                       weight=float(edge[2]) if len(edge) > 2 else 1.0)

    return graph


def main() -> int:
    out = {}

    for path in sorted(GRAPHS.glob("*.json")):
        if path.stem == "index":
            continue

        graph = load(path.stem)
        components = nx.number_connected_components(graph)

        minimum = list(nx.minimum_spanning_edges(graph, weight="weight", data=True))
        maximum = list(nx.maximum_spanning_edges(graph, weight="weight", data=True))

        out[path.stem] = {
            "nodes": graph.number_of_nodes(),
            "components": components,
            "edges_expected": graph.number_of_nodes() - components,
            "minimum": {
                "edges": len(minimum),
                "weight": sum(d["weight"] for _, _, d in minimum),
            },
            "maximum": {
                "edges": len(maximum),
                "weight": sum(d["weight"] for _, _, d in maximum),
            },
        }

    document = {
        "generator": "networkx.minimum_spanning_edges and maximum_spanning_edges",
        "regenerate": "python3 tools/generate_spanning_tree_fixtures.py",
        "note": (
            "Total weight and edge count, not the edge set. A spanning tree is not unique when "
            "weights tie, and these fixtures are unweighted or nearly so; pinning the edges "
            "would pin NetworkX's tie-breaking, which is not a property of the answer."
        ),
        "graphs": out,
    }

    path = ROOT / "resources/fixtures/spanning_tree.json"
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")

    print(f"-> {path}")

    for name, entry in out.items():
        print(f"   {name:22s} {entry['nodes']:3d} nodes, {entry['components']:2d} components, "
              f"{entry['edges_expected']:3d} edges  "
              f"min {entry['minimum']['weight']:8.2f}  max {entry['maximum']['weight']:8.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
