#!/usr/bin/env python3
"""
Reference bands for label propagation, from NetworkX.

Label propagation (Raghavan, Albert and Kumara, 2007) is the cheap one. Every
node starts as its own community and repeatedly adopts whichever label is
commonest among its neighbours; the whole thing runs in time linear in the
number of edges, with no objective function, no resolution parameter and
nothing to tune. It is what you reach for when the graph is too large for
Leiden to be worth the wait, or when you want a second opinion that shares
none of Leiden's assumptions.

The price is that it is unstable. There is no quantity being maximised, so
there is nothing to say one run was better than another, and on a graph
without clear structure it can collapse everything into one community or leave
everything separate. Both failures are real and are recorded here rather than
hidden: the band is what NetworkX produces over fifty seeds, and its width is
the honest description of the algorithm.

So this fixture pins a spread, not an answer, and the test that reads it asks
whether this implementation lands in the same territory -- not whether it
reproduces a particular run, which nothing reproduces.

Usage: python3 tools/generate_label_propagation_fixtures.py
"""

from __future__ import annotations

import json
import pathlib
import statistics

import networkx as nx

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPHS = ROOT / "resources/fixtures/graph"

SEEDS = list(range(50))


def load(name: str) -> nx.Graph:
    document = json.loads((GRAPHS / f"{name}.json").read_text())
    graph = nx.Graph()
    graph.add_nodes_from(range(document["nodes"]))

    for edge in document["edges"]:
        graph.add_edge(int(edge[0]), int(edge[1]),
                       weight=float(edge[2]) if len(edge) > 2 else 1.0)

    return graph


def band(graph: nx.Graph) -> dict:
    qualities, counts = [], []

    for seed in SEEDS:
        communities = list(nx.community.asyn_lpa_communities(graph, weight="weight", seed=seed))
        qualities.append(nx.community.modularity(graph, communities, weight="weight"))
        counts.append(len(communities))

    return {
        "seeds": len(SEEDS),
        "modularity": {
            "min": min(qualities),
            "max": max(qualities),
            "mean": statistics.fmean(qualities),
            "stdev": statistics.pstdev(qualities),
        },
        "communities": {"min": min(counts), "max": max(counts)},
    }


def main() -> int:
    names = sorted(p.stem for p in GRAPHS.glob("*.json") if p.stem != "index")
    out = {name: band(load(name)) for name in names}

    document = {
        "generator": "networkx.community.asyn_lpa_communities over 50 seeds",
        "regenerate": "python3 tools/generate_label_propagation_fixtures.py",
        "note": (
            "A spread, not an answer. Label propagation optimises nothing, so no run is better "
            "than another and none is reproducible across implementations; the width of the band "
            "is the honest description of the algorithm rather than a defect of the fixture."
        ),
        "graphs": out,
    }

    path = ROOT / "resources/fixtures/label_propagation.json"
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")

    print(f"-> {path}")

    for name, entry in out.items():
        print(f"   {name:22s} Q {entry['modularity']['min']:7.4f}..{entry['modularity']['max']:7.4f}"
              f"   communities {entry['communities']['min']:3d}..{entry['communities']['max']:3d}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
