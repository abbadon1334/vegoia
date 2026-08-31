#!/usr/bin/env python3
"""
Reference values for the structural questions, from NetworkX.

Three things a graph library is expected to answer and this one could not.

Strongly connected components, on directed graphs. The undirected notion --
can I get from here to there -- splits into two on a directed graph, and only
the strong one is an equivalence relation. It is what tells you whether a
citation graph has genuine cycles or only looks like it, and what a topological
sort needs to exist.

Bridges and articulation points, on undirected graphs. A bridge is an edge
whose removal disconnects the graph; an articulation point is a node whose
removal does. Together they are where a graph is fragile, which for a
knowledge graph is where a single extracted relation is carrying an entire
region of the answer -- exactly the edge you want a human to check.

Every fixture in the repository is used, directed ones for the first and
undirected ones for the other two, so the reference covers the shapes that are
already known to be awkward: the bow tie, the disjoint cliques, the ring of
cliques whose bridges are the ring.

Usage: python3 tools/generate_structure_fixtures.py
"""

from __future__ import annotations

import json
import pathlib

import networkx as nx

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPHS = ROOT / "resources/fixtures/graph"
DIRECTED = ROOT / "resources/fixtures/directed"


def build(path: pathlib.Path, directed: bool):
    document = json.loads(path.read_text())
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(range(document["nodes"]))

    for edge in document["edges"]:
        graph.add_edge(int(edge[0]), int(edge[1]))

    return graph


def canonical(groups) -> list[list[int]]:
    """Components as sorted lists, ordered by their smallest member.

    Neither the order of the components nor of their members carries meaning,
    so both are fixed here rather than left to whichever traversal produced
    them -- otherwise the fixture would pin NetworkX's iteration order and
    fail on any implementation that walked the graph differently.
    """
    return sorted((sorted(int(v) for v in group) for group in groups), key=lambda g: g[0])


def main() -> int:
    strong = {}

    for path in sorted(DIRECTED.glob("*.json")):
        if path.name == "index.json":
            continue

        graph = build(path, directed=True)
        strong[path.stem] = {
            "components": canonical(nx.strongly_connected_components(graph)),
            "weak_components": canonical(nx.weakly_connected_components(graph)),
            "is_strongly_connected": bool(
                graph.number_of_nodes() > 0 and nx.is_strongly_connected(graph)
            ),
            "is_directed_acyclic": bool(nx.is_directed_acyclic_graph(graph)),
        }

    fragile = {}

    for path in sorted(GRAPHS.glob("*.json")):
        if path.name == "index.json":
            continue

        graph = build(path, directed=False)
        fragile[path.stem] = {
            # Edges as sorted pairs, sorted, for the same reason components are.
            "bridges": sorted([min(u, v), max(u, v)] for u, v in nx.bridges(graph)),
            "articulation_points": sorted(int(v) for v in nx.articulation_points(graph)),
        }

    document = {
        "generator": "networkx: strongly_connected_components, weakly_connected_components, "
                     "is_strongly_connected, is_directed_acyclic_graph, bridges, articulation_points",
        "regenerate": "python3 tools/generate_structure_fixtures.py",
        "note": (
            "Components are sorted within and between, and bridges are sorted pairs sorted "
            "again, because neither order carries meaning and pinning NetworkX's traversal "
            "order would fail any implementation that walked the graph differently."
        ),
        "directed": strong,
        "undirected": fragile,
    }

    path = ROOT / "resources/fixtures/structure.json"
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")

    print(f"-> {path}")

    for name, entry in strong.items():
        print(f"   {name:22s} {len(entry['components']):3d} strong, "
              f"{len(entry['weak_components']):3d} weak"
              f"{', acyclic' if entry['is_directed_acyclic'] else ''}")

    for name, entry in fragile.items():
        print(f"   {name:22s} {len(entry['bridges']):3d} bridges, "
              f"{len(entry['articulation_points']):3d} articulation points")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
