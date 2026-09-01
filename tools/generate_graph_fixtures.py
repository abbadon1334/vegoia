#!/usr/bin/env python3
"""
Generate golden fixtures for the graph side of Vegoia.

The oracle is the reference implementation, never our own code:

  * community detection -> `leidenalg` by V.A. Traag, the author of the
    Leiden algorithm (Traag, Waltman & van Eck, 2019).
  * centrality / paths  -> `networkx`, whose conventions are the ones most
    widely documented and cross-checked.

Leiden is stochastic. Committing one partition as "the" answer would be
wrong: our PHP implementation cannot reproduce Python's RNG stream. So for
each graph we record the *distribution* of the quality score over many
seeds. That turns an arbitrary tolerance into a measured one -- a PHP run
is acceptable when it lands inside the envelope the reference itself
occupies.

Usage:  python3 tools/generate_graph_fixtures.py [outdir]
"""

from __future__ import annotations

import json
import os
import pathlib
import statistics
import sys

import igraph as ig
import leidenalg
import networkx as nx

# Python randomises string hashing per process, which changes the iteration
# order of sets of node names, which changes the order floating-point sums are
# accumulated in -- and that moves the last bit of results like harmonic
# centrality. The fixtures then differ between regenerations for no reason
# anyone would guess. Re-exec once with the seed pinned so the output is a
# function of the input alone.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])



SEEDS = list(range(1, 51))


def weights_of(g: ig.Graph):
    """igraph silently ignores edge weights unless you name the attribute.
    Every call that can take weights must therefore be told about them, or the
    fixture records the unweighted answer for a weighted graph -- which looks
    plausible and is wrong."""
    return "weight" if "weight" in g.es.attributes() else None


def envelope(g: ig.Graph, partition_type, **kwargs) -> dict:
    """Run the reference implementation across many seeds, record the spread."""
    w = weights_of(g)
    qs, ks = [], []
    for seed in SEEDS:
        p = leidenalg.find_partition(g, partition_type, seed=seed, weights=w, **kwargs)
        qs.append(g.modularity(p.membership, weights=w))
        ks.append(len(set(p.membership)))
    return {
        "seeds": len(SEEDS),
        "modularity": {
            "min": min(qs),
            "max": max(qs),
            "mean": statistics.fmean(qs),
            "stdev": statistics.pstdev(qs),
        },
        "communities": {"min": min(ks), "max": max(ks)},
    }


def node_order(G: nx.Graph) -> dict:
    """Relabel nodes to 0..n-1, stably.

    networkx graphs are keyed by whatever the generator chose -- ints, names,
    coordinate tuples -- so some canonical order is needed. Sorting everything
    as a string would be simplest, but it puts node 10 before node 2, which
    makes an integer-labelled fixture unreadable and its expected partitions
    impossible to write down by hand. Integers therefore sort as integers.
    """
    nodes = list(G.nodes())

    if all(isinstance(v, int) for v in nodes):
        return {v: i for i, v in enumerate(sorted(nodes))}

    return {v: i for i, v in enumerate(sorted(nodes, key=str))}


def from_networkx(G: nx.Graph) -> tuple[ig.Graph, list[list]]:
    """Relabel to 0..n-1 in a stable order, return (igraph, edge list)."""
    order = node_order(G)
    edges = sorted(
        [order[u], order[v], float(d.get("weight", 1.0))]
        for u, v, d in G.edges(data=True)
    )
    g = ig.Graph(n=G.number_of_nodes(), edges=[(e[0], e[1]) for e in edges])
    if any(e[2] != 1.0 for e in edges):
        g.es["weight"] = [e[2] for e in edges]
    return g, edges


def katz_alpha(G: nx.Graph, weighted: bool) -> float:
    """Half of 1 / lambda_max: safely convergent, and not so small that every
    node collapses onto the beta baseline."""
    import numpy as np
    A = nx.to_numpy_array(G, weight="weight" if weighted else None)
    return 0.5 / float(max(abs(np.linalg.eigvalsh(A))))


def centralities(G: nx.Graph, n: int) -> dict:
    """Reference centralities. Conventions are pinned and documented here."""
    weighted = any("weight" in d for _, _, d in G.edges(data=True))
    key = node_order(G)

    def vec(d, default=0.0):
        out = [default] * n
        for node, value in d.items():
            out[key[node]] = float(value)
        return out

    out = {
        # power iteration, alpha=0.85, dangling mass spread uniformly, sums to 1
        "pagerank": vec(nx.pagerank(G, alpha=0.85, tol=1.0e-12, max_iter=1000)),
        # L2-normalised principal eigenvector of the adjacency matrix.
        # weight is passed explicitly: networkx defaults eigenvector_centrality
        # to weight=None while defaulting pagerank to weight="weight", an
        # inconsistency on their side. Vegoia uses weights throughout, so the
        # fixture is generated the weighted way.
        "eigenvector": vec(nx.eigenvector_centrality(
            G, max_iter=10000, tol=1.0e-12, weight="weight" if weighted else None)),
        # sum of 1/d over reachable nodes; unreachable contribute nothing
        "harmonic": vec(nx.harmonic_centrality(G)),
        # Katz needs alpha below 1/lambda_max or the series diverges -- longer
        # walks would outweigh shorter ones, which is not a centrality. A fixed
        # alpha cannot serve every graph: on the weighted ones lambda_max is
        # large enough that 0.05 already diverges. Half the critical value is
        # used instead, and recorded, so the fixture is meaningful per graph.
        "katz": vec(nx.katz_centrality_numpy(
            G, alpha=katz_alpha(G, weighted), beta=1.0,
            weight="weight" if weighted else None)),
        # local clustering coefficient
        "clustering": vec(nx.clustering(G)),
        # triangles through each node
        "triangles": vec(nx.triangles(G)),
        # k-core number
        "core_number": vec(nx.core_number(G)),
        # Brandes, NOT normalised; undirected pair counted once
        "betweenness": vec(nx.betweenness_centrality(G, normalized=False)),
        # Wasserman-Faust correction, so disconnected graphs stay meaningful
        "closeness": vec(nx.closeness_centrality(G, wf_improved=True)),
        "degree": vec(dict(G.degree())),
    }
    if weighted:
        out["betweenness_weighted"] = vec(
            nx.betweenness_centrality(G, normalized=False, weight="weight")
        )
    # Personalised PageRank: the teleport target is a chosen set rather than
    # the whole graph, which is how a ranking is biased towards a query's seed
    # nodes. Three seeds spread across the graph so the effect is visible.
    seeds = sorted(key)[:3]
    personal = {v: (1.0 if key[v] in (0, n // 2, n - 1) else 0.0) for v in G}
    if sum(personal.values()) == 0:
        personal = {v: 1.0 for v in G}
    out["pagerank_personalised"] = vec(
        nx.pagerank(G, alpha=0.85, tol=1.0e-12, max_iter=1000, personalization=personal))
    out["personalisation_nodes"] = [float(x) for x in sorted({0, n // 2, n - 1})]

    # Betweenness with weights read as distances, not capacities: a heavier
    # edge is a longer detour, so shortest paths avoid it.
    out["betweenness_weighted"] = vec(
        nx.betweenness_centrality(G, normalized=False, weight="weight" if weighted else None))

    out["katz_alpha"] = [katz_alpha(G, weighted)] * n
    return out


def shortest_paths(G: nx.Graph, n: int) -> dict:
    """All-pairs distances from source 0; -1 means unreachable."""
    key = node_order(G)
    inv = {i: v for v, i in key.items()}
    unweighted = dict(nx.single_source_shortest_path_length(G, inv[0]))
    row = [-1.0] * n
    for node, d in unweighted.items():
        row[key[node]] = float(d)
    out = {"bfs_from_0": row}

    # Emitted for every graph, weighted or not. On an unweighted graph the two
    # rows are equal -- NetworkX treats a missing weight as 1 -- and that is
    # the reason to record it rather than the reason to skip it: "Dijkstra
    # reduces to BFS here" is a statement about the mathematics, not about
    # whether this implementation of Dijkstra does. HITS was wrong for months
    # behind exactly that kind of reasoning.
    w = dict(nx.single_source_dijkstra_path_length(G, inv[0], weight="weight"))
    wrow = [-1.0] * n
    for node, d in w.items():
        wrow[key[node]] = float(d)
    out["dijkstra_from_0"] = wrow

    return out


def quality_probes(g: ig.Graph) -> dict:
    """Deterministic checkpoints for the quality functions themselves.

    Leiden is stochastic, so its output can only be tested against an envelope.
    Modularity and CPM are not: given a graph and a partition they are plain
    arithmetic, and igraph will agree to the last bit. Pinning a few fixed
    partitions here means a bug in the quality function is caught exactly,
    instead of hiding inside a tolerance meant for the search.
    """
    n = g.vcount()
    singletons = list(range(n))
    single = [0] * n
    w = weights_of(g)
    reference = leidenalg.find_partition(
        g, leidenalg.ModularityVertexPartition, seed=42, weights=w
    ).membership

    def cpm(membership: list[int], resolution: float) -> float:
        """CPM as leidenalg defines it: sum over communities of
        (internal weight - resolution * n_c(n_c-1)/2)."""
        total = 0.0
        for c in set(membership):
            members = [v for v, m in enumerate(membership) if m == c]
            sub = g.subgraph(members)
            w = (sum(sub.es["weight"]) if "weight" in sub.es.attributes()
                 else sub.ecount())
            k = len(members)
            total += w - resolution * k * (k - 1) / 2
        return total

    def quality_of(partition_class, membership, **kwargs):
        """leidenalg reports UNNORMALISED quality: modularity times 2m, and so
        on. Recorded raw so the PHP side can be compared on its own terms after
        the documented rescaling, rather than through a conversion buried in a
        test."""
        try:
            p = partition_class(g, initial_membership=list(membership), **kwargs)
            return p.quality()
        except Exception as exc:                       # pragma: no cover
            return {"error": str(exc)}

    probes = {}
    for label, membership in (("singletons", singletons),
                              ("single_community", single),
                              ("leiden_seed42", reference)):
        probes[label] = {
            "membership": list(membership),
            "modularity": {
                str(r): g.modularity(membership, resolution=r, weights=w)
                for r in (0.5, 1.0, 2.0)
            },
            "cpm": {str(r): cpm(list(membership), r) for r in (0.05, 0.5, 1.0)},
            # Reichardt-Bornholdt with an Erdos-Renyi null model: like CPM but
            # with the resolution scaled by the graph's own density.
            "rber": {
                str(r): quality_of(leidenalg.RBERVertexPartition, membership,
                                   weights=w, resolution_parameter=r)
                for r in (0.5, 1.0, 2.0)
            },
            # Surprise and Significance are defined on unweighted graphs only;
            # leidenalg takes no weights for them.
            "surprise": quality_of(leidenalg.SurpriseVertexPartition, membership),
            "significance": quality_of(leidenalg.SignificanceVertexPartition, membership),
        }
    return probes


def build(name: str, G: nx.Graph, source: str, note: str = "") -> dict:
    g, edges = from_networkx(G)
    n = G.number_of_nodes()
    fixture = {
        "name": name,
        "source": source,
        "note": note,
        "directed": False,
        "nodes": n,
        "edges": edges,
        "expected": {
            "components": nx.number_connected_components(G),
            "density": nx.density(G),
            "transitivity": nx.transitivity(G),
            "average_clustering": nx.average_clustering(G),
            "triangle_total": sum(nx.triangles(G).values()) // 3,
            "centrality": centralities(G, n),
            "paths": shortest_paths(G, n),
            "quality_probes": quality_probes(g),
            "leiden": {
                "modularity_objective": envelope(g, leidenalg.ModularityVertexPartition),
                "cpm": {
                    str(r): envelope(
                        g, leidenalg.CPMVertexPartition, resolution_parameter=r
                    )
                    for r in (0.05, 0.1, 0.5)
                },
            },
        },
    }
    return fixture


def ring_of_cliques(k: int, s: int) -> nx.Graph:
    """k cliques of size s joined in a ring: the resolution-limit counterexample
    of Fortunato & Barthelemy (2007). The ground truth is unambiguous -- the
    cliques -- yet modularity maximisation merges adjacent pairs once k grows."""
    G = nx.Graph()
    for c in range(k):
        base = c * s
        for i in range(s):
            for j in range(i + 1, s):
                G.add_edge(base + i, base + j)
    for c in range(k):
        G.add_edge(c * s, ((c + 1) % k) * s + 1)
    return G


def disjoint_cliques(sizes: list[int]) -> nx.Graph:
    """Fully separated cliques. The only defensible partition is the obvious
    one, so this is an exact-equality test rather than an envelope test."""
    G = nx.Graph()
    base = 0
    for s in sizes:
        for i in range(s):
            for j in range(i + 1, s):
                G.add_edge(base + i, base + j)
        base += s
    return G


def main() -> int:
    outdir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "resources/fixtures/graph")
    outdir.mkdir(parents=True, exist_ok=True)

    catalogue = [
        ("zachary", nx.karate_club_graph(), "networkx.karate_club_graph()",
         "Zachary 1977. The canonical community-detection benchmark; max known modularity 0.4198."),
        ("florentine", nx.florentine_families_graph(), "networkx.florentine_families_graph()",
         "Padgett & Ansell 1993. Small and disconnected-adjacent: good for edge cases."),
        ("davis", nx.davis_southern_women_graph(), "networkx.davis_southern_women_graph()",
         "Davis 1941. Bipartite, so community structure is a known trap."),
        ("lesmis", nx.les_miserables_graph(), "networkx.les_miserables_graph()",
         "Knuth 1993. Weighted co-appearance graph; exercises the weighted paths."),
        ("petersen", nx.petersen_graph(), "networkx.petersen_graph()",
         "Vertex-transitive: every node has identical centrality, so any asymmetry is a bug."),
        ("star_10", nx.star_graph(9), "networkx.star_graph(9)",
         "Closed form: hub betweenness = (n-1)(n-2)/2 = 36. Checkable by hand."),
        ("path_10", nx.path_graph(10), "networkx.path_graph(10)",
         "Closed form for distances and betweenness. Checkable by hand."),
        ("complete_8", nx.complete_graph(8), "networkx.complete_graph(8)",
         "All betweenness zero, all distances 1. Degenerate on purpose."),
        ("grid_4x4", nx.grid_2d_graph(4, 4), "networkx.grid_2d_graph(4, 4)",
         "Regular lattice with no community structure: guards against inventing one."),
        ("ring_of_cliques_10x5", ring_of_cliques(10, 5), "vegoia:ring_of_cliques(10, 5)",
         "Fortunato & Barthelemy 2007 resolution limit. Ground truth is the 10 cliques."),
        ("disjoint_cliques", disjoint_cliques([5, 4, 3]), "vegoia:disjoint_cliques([5,4,3])",
         "Three separated cliques. Exact expected partition, not an envelope."),
    ]

    index = []
    for name, G, source, note in catalogue:
        fx = build(name, G, source, note)
        path = outdir / f"{name}.json"
        path.write_text(json.dumps(fx, indent=1, sort_keys=True) + "\n")
        q = fx["expected"]["leiden"]["modularity_objective"]["modularity"]
        print(
            f"{name:22s} n={fx['nodes']:4d} m={len(fx['edges']):5d} "
            f"Q=[{q['min']:.6f}, {q['max']:.6f}] "
            f"k={fx['expected']['leiden']['modularity_objective']['communities']}"
        )
        index.append({"name": name, "file": f"{name}.json", "nodes": fx["nodes"],
                      "edges": len(fx["edges"]), "note": note})

    (outdir / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    print(f"\n{len(index)} fixtures -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
