#!/usr/bin/env python3
"""
Reference values for link prediction, from NetworkX.

Link prediction scores a pair of nodes that are not joined by how plausible an
edge between them would be. That is the question a retrieval-augmented system
asks constantly and rarely names: given the entity I just found, which other
entities is it probably related to even though nobody wrote the edge down?

Five measures, and they disagree on purpose:

  * common neighbours, the count and nothing else -- the baseline everything
    is measured against;
  * Jaccard, that count divided by the size of the union, so a pair of
    low-degree nodes sharing two neighbours beats a pair of hubs sharing three;
  * Adamic-Adar, which weights each shared neighbour by 1/log(degree), so
    sharing an obscure neighbour counts for more than sharing a hub;
  * resource allocation, the same idea with 1/degree, which punishes hubs
    harder still;
  * preferential attachment, the product of the degrees, which ignores shared
    neighbours entirely and says only that busy nodes get busier.

Scored on every pair of the small fixtures, joined or not: an implementation
that quietly assumed the pair was non-adjacent would pass a test that only
used the pairs it was designed for.

Usage: python3 tools/generate_link_prediction_fixtures.py
"""

from __future__ import annotations

import itertools
import json
import pathlib

import networkx as nx

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPHS = ROOT / "resources/fixtures/graph"

# The whole collection would be 100k pairs on the larger fixtures for no extra
# information; these span dense, sparse, bipartite and regular.
CHOSEN = ["zachary", "florentine", "davis", "petersen", "path_10", "grid_4x4",
          "star_10", "complete_8", "disjoint_cliques"]


def load(name: str) -> tuple[nx.Graph, list[list[float]]]:
    document = json.loads((GRAPHS / f"{name}.json").read_text())
    graph = nx.Graph()
    graph.add_nodes_from(range(document["nodes"]))

    for edge in document["edges"]:
        graph.add_edge(int(edge[0]), int(edge[1]))

    return graph, document["edges"]


def measures(graph: nx.Graph, pairs: list[tuple[int, int]]) -> dict:
    def scores(function) -> dict:
        return {f"{u},{v}": float(s) for u, v, s in function(graph, pairs)}

    return {
        "common_neighbours": {
            f"{u},{v}": len(list(nx.common_neighbors(graph, u, v))) for u, v in pairs
        },
        "jaccard": scores(nx.jaccard_coefficient),
        "adamic_adar": scores(nx.adamic_adar_index),
        "resource_allocation": scores(nx.resource_allocation_index),
        "preferential_attachment": scores(nx.preferential_attachment),
    }


def main() -> int:
    out = {}

    for name in CHOSEN:
        graph, _ = load(name)
        pairs = [(u, v) for u, v in itertools.combinations(sorted(graph.nodes), 2)]

        out[name] = {
            "nodes": graph.number_of_nodes(),
            "pairs": len(pairs),
            "adjacent": sorted(f"{u},{v}" for u, v in pairs if graph.has_edge(u, v)),
            "measures": measures(graph, pairs),
        }

    document = {
        "generator": "networkx: jaccard_coefficient, adamic_adar_index, "
                     "resource_allocation_index, preferential_attachment, common_neighbors",
        "regenerate": "python3 tools/generate_link_prediction_fixtures.py",
        "note": (
            "Every unordered pair of every listed graph, adjacent ones included. NetworkX defines "
            "all five on any pair, adjacency being irrelevant to the arithmetic, and pinning both "
            "kinds is what stops an implementation quietly assuming one."
        ),
        "graphs": out,
    }

    path = ROOT / "resources/fixtures/link_prediction.json"
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")

    print(f"-> {path}")

    for name, entry in out.items():
        print(f"   {name:20s} {entry['nodes']:4d} nodes, {entry['pairs']:5d} pairs, "
              f"{len(entry['adjacent']):4d} of them joined")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
