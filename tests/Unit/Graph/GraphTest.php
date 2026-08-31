<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit\Graph;

use function iterator_to_array;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Graph\Graph;
use Vegoia\Tests\Support\GraphFixture;

#[CoversClass(Graph::class)]
final class GraphTest extends TestCase
{
    public function test_it_reports_order_and_size(): void
    {
        $graph = Graph::undirected(4, [[0, 1, 1.0], [1, 2, 1.0], [2, 3, 1.0]]);

        self::assertSame(4, $graph->order());
        self::assertSame(3, $graph->size());
        self::assertFalse($graph->isDirected());
    }

    public function test_an_undirected_edge_is_visible_from_both_ends(): void
    {
        $graph = Graph::undirected(2, [[0, 1, 2.5]]);

        self::assertSame([1 => 2.5], iterator_to_array($graph->neighbours(0)));
        self::assertSame([0 => 2.5], iterator_to_array($graph->neighbours(1)));
        self::assertTrue($graph->hasEdge(1, 0));
        self::assertSame(2.5, $graph->edgeWeight(1, 0));
    }

    public function test_degree_counts_edges_and_strength_sums_weights(): void
    {
        $graph = Graph::undirected(3, [[0, 1, 2.0], [0, 2, 3.0]]);

        self::assertSame(2, $graph->degree(0));
        self::assertSame(5.0, $graph->strength(0));
        self::assertSame(1, $graph->degree(1));
        self::assertSame(2.0, $graph->strength(1));
    }

    /**
     * The convention matters and it is not arbitrary: modularity is derived
     * from the degree sum, and igraph -- the reference our fixtures come from
     * -- counts an undirected self-loop twice. Diverging here would make every
     * modularity comparison wrong by a term nobody would think to look at.
     */
    public function test_a_self_loop_counts_twice_towards_degree_and_strength(): void
    {
        $graph = Graph::undirected(2, [[0, 0, 1.5], [0, 1, 1.0]]);

        self::assertSame(3, $graph->degree(0), 'self-loop contributes 2 to the degree');
        self::assertSame(4.0, $graph->strength(0), 'self-loop contributes 2w to the strength');
        self::assertSame(1.5, $graph->selfLoopWeight(0));
        self::assertSame(0.0, $graph->selfLoopWeight(1));
    }

    public function test_parallel_edges_are_merged_by_summing_their_weights(): void
    {
        $graph = Graph::undirected(2, [[0, 1, 1.0], [1, 0, 2.0], [0, 1, 0.5]]);

        self::assertSame(1, $graph->size(), 'three inputs describe one edge');
        self::assertSame(3.5, $graph->edgeWeight(0, 1));
    }

    public function test_total_weight_is_the_sum_over_edges_not_over_endpoints(): void
    {
        $graph = Graph::undirected(3, [[0, 1, 2.0], [1, 2, 3.0]]);

        self::assertSame(5.0, $graph->totalWeight());
        // 2m in the modularity literature: the degree sum.
        self::assertSame(10.0, $graph->totalEndpointWeight());
    }

    /**
     * Describes the graph, not the input. Two parallel edges of weight 1 merge
     * into one of weight 2, so a construction with no explicit weights can
     * still be a weighted graph -- and reporting the input gave two identical
     * graphs different answers.
     */
    public function test_weightedness_describes_the_stored_graph(): void
    {
        self::assertFalse(Graph::undirected(2, [[0, 1, 1.0]])->isWeighted());
        self::assertTrue(Graph::undirected(2, [[0, 1, 2.0]])->isWeighted());

        $merged = Graph::undirected(2, [[0, 1, 1.0], [0, 1, 1.0]]);

        self::assertSame(2.0, $merged->edgeWeight(0, 1));
        self::assertTrue($merged->isWeighted(), 'merging made it weighted');

        self::assertFalse(Graph::undirected(3)->isWeighted(), 'no edges, no weights');
    }

    public function test_it_enumerates_its_nodes(): void
    {
        self::assertSame([0, 1, 2], iterator_to_array(Graph::undirected(3)->nodes()));
        self::assertSame([], iterator_to_array(Graph::undirected(0)->nodes()));
    }

    public function test_it_rejects_a_node_outside_the_declared_order(): void
    {
        $this->expectException(InvalidArgument::class);

        Graph::undirected(2, [[0, 5, 1.0]]);
    }

    public function test_it_rejects_querying_a_node_that_does_not_exist(): void
    {
        $graph = Graph::undirected(2, [[0, 1, 1.0]]);

        $this->expectException(InvalidArgument::class);

        $graph->degree(7);
    }

    public function test_an_isolated_node_is_still_a_node(): void
    {
        $graph = Graph::undirected(5, [[0, 1, 1.0]]);

        self::assertSame(5, $graph->order());
        self::assertSame(0, $graph->degree(4));
        self::assertSame([], iterator_to_array($graph->neighbours(4)));
    }

    public function test_edges_are_enumerated_once_each(): void
    {
        $graph = Graph::undirected(3, [[0, 1, 1.0], [1, 2, 2.0]]);

        $seen = [];
        foreach ($graph->edges() as [$u, $v, $w]) {
            $seen[] = [$u, $v, $w];
        }

        self::assertSame([[0, 1, 1.0], [1, 2, 2.0]], $seen);
    }

    /** @return iterable<string, array{string}> */
    public static function fixtures(): iterable
    {
        foreach (GraphFixture::names() as $name) {
            yield $name => [$name];
        }
    }

    #[DataProvider('fixtures')]
    public function test_degrees_agree_with_networkx_on_every_fixture(string $name): void
    {
        $fixture = GraphFixture::load($name);
        $graph = $fixture->graph();

        self::assertSame($fixture->nodes, $graph->order());

        $expected = $fixture->expectedVector('degree');

        for ($node = 0; $node < $graph->order(); $node++) {
            self::assertSame(
                (int) $expected[$node],
                $graph->degree($node),
                "{$name}: degree of node {$node}",
            );
        }
    }
}
