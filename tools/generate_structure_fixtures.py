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

import warnings

import networkx as nx
import numpy as np

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


def distance(graph: nx.Graph) -> dict:
    """Eccentricity within each node's own component, and the summaries.

    Computed per component because NetworkX refuses the whole graph -- diameter,
    radius and average_shortest_path_length all raise NetworkXError when it is
    disconnected. That refusal is about their definitions on the graph as a
    whole, not about the numbers being unavailable: a caller wanting them would
    loop the components exactly as this does. Two of the eleven fixtures are
    disconnected, one of them the largest and most realistic, so a reference
    that stopped at the exception would have nothing to say about either.

    Eccentricity is therefore always finite. The summaries are recorded twice:
    for the whole graph, where a disconnected one gives infinity because some
    pair genuinely has no path, and for the largest component, which is what
    anybody actually looks at.
    """
    order = graph.number_of_nodes()

    if order == 0:
        return {"eccentricity": [], "components": 0, "total_distance": 0.0,
                "reachable_pairs": 0, "diameter": 0.0, "radius": 0.0,
                "average_shortest_path_length": 0.0, "largest_component": None}

    components = list(nx.connected_components(graph))
    eccentricity = [0.0] * order
    total = 0
    pairs = 0
    largest = max(components, key=len)
    largest_summary = None

    for members in components:
        sub = graph.subgraph(members)
        ecc = nx.eccentricity(sub)

        for node, value in ecc.items():
            eccentricity[int(node)] = float(value)

        for source, lengths in nx.all_pairs_shortest_path_length(sub):
            for target, d in lengths.items():
                if source != target:
                    total += d
                    pairs += 1

        if members is largest or set(members) == set(largest):
            largest_summary = {
                "nodes": len(members),
                "diameter": float(nx.diameter(sub)),
                "radius": float(nx.radius(sub)),
                "average_shortest_path_length": float(nx.average_shortest_path_length(sub)),
            }

    connected = len(components) == 1

    return {
        "eccentricity": eccentricity,
        "components": len(components),
        "total_distance": float(total),
        "reachable_pairs": pairs,
        # INF on the whole graph when it is disconnected: some pair has no
        # path, so the maximum over pairs is infinite and so is the minimum
        # eccentricity, since every node fails to reach something.
        "diameter": float(nx.diameter(graph)) if connected else None,
        "radius": float(nx.radius(graph)) if connected else None,
        "average_shortest_path_length":
            float(nx.average_shortest_path_length(graph)) if connected else None,
        "largest_component": largest_summary,
    }


def assortativity(graph: nx.Graph) -> float | None:
    """Newman's degree assortativity, or null where it is undefined.

    On a regular graph every edge joins two nodes of the same degree, so the
    correlation is 0/0. NetworkX returns nan with a warning; null here, and the
    PHP side refuses -- which is what Correlation::pearson already does for a
    sample with no variation, and this is literally that.
    """
    if graph.number_of_edges() == 0:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = nx.degree_assortativity_coefficient(graph)

    return None if np.isnan(value) else float(value)


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
            "distance": distance(graph),
            "degree_assortativity": assortativity(graph),
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
        d = entry["distance"]
        diameter = "INF" if d["diameter"] is None else f"{d['diameter']:.0f}"
        assort = "undefined" if entry["degree_assortativity"] is None \
            else f"{entry['degree_assortativity']:8.5f}"
        print(f"   {name:22s} {len(entry['bridges']):3d} bridges, "
              f"{len(entry['articulation_points']):3d} cut vertices, "
              f"diam {diameter:>3s}, assortativity {assort}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
