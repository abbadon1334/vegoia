<?php

declare(strict_types=1);

namespace Vegoia\Tests\Reference\Graph;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Graph\Centrality\Betweenness;
use Vegoia\Graph\Centrality\PageRank;
use Vegoia\Graph\Community\Leiden;
use Vegoia\Graph\Graph;
use Vegoia\Graph\Path\BreadthFirst;
use Vegoia\Graph\Path\Dijkstra;

/**
 * Directed graphs.
 *
 * Graph::directed() was public API with no test touching it at all -- found by
 * auditing coverage against networkx rather than by anything failing. Untested
 * public API is worse than absent API: it is a promise nobody checked.
 *
 * Expected values are networkx's on the same graphs.
 */
#[CoversClass(Graph::class)]
#[Group('reference')]
final class DirectedGraphTest extends TestCase
{
    public function test_an_edge_is_visible_only_in_its_own_direction(): void
    {
        $graph = Graph::directed(3, [[0, 1, 1.0], [1, 2, 1.0]]);

        self::assertTrue($graph->isDirected());
        self::assertSame(2, $graph->size());
        self::assertTrue($graph->hasEdge(0, 1));
        self::assertFalse($graph->hasEdge(1, 0));
        self::assertSame(1.0, $graph->edgeWeight(0, 1));
        self::assertSame(0.0, $graph->edgeWeight(1, 0));
    }

    /** Degree and strength count outgoing edges: the CSR row is the out-neighbourhood. */
    public function test_degree_counts_only_outgoing_edges(): void
    {
        $graph = Graph::directed(3, [[0, 1, 2.0], [1, 2, 3.0]]);

        self::assertSame(1, $graph->degree(0));
        self::assertSame(2.0, $graph->strength(0));
        self::assertSame(0, $graph->degree(2), 'a sink has no outgoing edges');
        self::assertSame(0.0, $graph->strength(2));
    }

    /** Undirected halves the degree sum; directed does not. */
    public function test_total_endpoint_weight_is_not_doubled(): void
    {
        $graph = Graph::directed(3, [[0, 1, 2.0], [1, 2, 3.0]]);

        self::assertSame(5.0, $graph->totalWeight());
        self::assertSame(5.0, $graph->totalEndpointWeight());
    }

    public function test_opposite_edges_stay_separate_instead_of_merging(): void
    {
        $graph = Graph::directed(2, [[0, 1, 1.0], [1, 0, 4.0]]);

        self::assertSame(2, $graph->size(), 'a-b and b-a are two directed edges');
        self::assertSame(1.0, $graph->edgeWeight(0, 1));
        self::assertSame(4.0, $graph->edgeWeight(1, 0));
    }

    public function test_traversal_respects_direction(): void
    {
        $graph = Graph::directed(3, [[0, 1, 1.0], [1, 2, 1.0]]);

        self::assertSame([0.0, 1.0, 2.0], BreadthFirst::distancesFrom($graph, 0));
        self::assertSame([-1.0, -1.0, 0.0], BreadthFirst::distancesFrom($graph, 2), 'a sink reaches nobody');
        self::assertSame([-1.0, 0.0, 1.0], Dijkstra::distancesFrom($graph, 1));
    }

    /** networkx: nx.pagerank(DiGraph([(0,1),(1,2)]), alpha=0.85) */
    public function test_pagerank_matches_networkx_on_a_directed_chain(): void
    {
        $scores = (new PageRank(tolerance: 1.0e-14))->of(Graph::directed(3, [[0, 1, 1.0], [1, 2, 1.0]]));

        self::assertEqualsWithDelta(0.184417, $scores[0], 1.0e-6);
        self::assertEqualsWithDelta(0.341171, $scores[1], 1.0e-6);
        self::assertEqualsWithDelta(0.474412, $scores[2], 1.0e-6);
        self::assertEqualsWithDelta(1.0, array_sum($scores), 1.0e-12);
    }

    /** networkx: betweenness_centrality(DiGraph cycle, normalized=False) -> 3 each. */
    public function test_betweenness_matches_networkx_and_is_not_halved(): void
    {
        $graph = Graph::directed(4, [[0, 1, 1.0], [1, 2, 1.0], [2, 3, 1.0], [3, 0, 1.0]]);

        foreach (Betweenness::of($graph) as $node => $score) {
            self::assertSame(3.0, $score, "node {$node}: directed pairs are counted once");
        }
    }

    /**
     * Modularity on a directed graph is the Leicht-Newman formula, which is
     * not implemented. Running the undirected one anyway returned a partition
     * that looked entirely reasonable, which is the worst outcome available.
     */
    public function test_leiden_refuses_a_directed_graph_rather_than_answering_wrongly(): void
    {
        $this->expectException(InvalidArgument::class);
        $this->expectExceptionMessage('undirected graphs only');

        Leiden::modularity(seed: 1)->partition(Graph::directed(3, [[0, 1, 1.0], [1, 2, 1.0]]));
    }
}
