<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit\Interop;

use Fhaculty\Graph\Graph as ExternalGraph;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\TestCase;
use Vegoia\Graph\Community\Leiden;
use Vegoia\Interop\Graphp;

/**
 * Reading a graph built with graphp/graph.
 *
 * That library is a development dependency and stays one, so every test here
 * skips when it is absent rather than failing -- the point of the adapter is
 * that a program which never calls it never needs the package.
 */
#[CoversClass(Graphp::class)]
final class GraphpTest extends TestCase
{
    protected function setUp(): void
    {
        if (! Graphp::isAvailable()) {
            self::markTestSkipped('graphp/graph is not installed');
        }
    }

    /**
     * graphp annotates createVertex() as taking int|null and accepts strings
     * perfectly well -- its own documentation shows named vertices. The
     * annotation is narrower than the behaviour, so the ignores below are
     * about its types rather than about this code, and using integer ids
     * instead would test something other than what an adapter is for.
     */
    private static function build(): ExternalGraph
    {
        $graph = new ExternalGraph();

        foreach (['alice', 'bob', 'carol', 'dave'] as $id) {
            /** @phpstan-ignore argument.type */
            $graph->createVertex($id);
        }

        $graph->getVertex('alice')->createEdge($graph->getVertex('bob'));
        $graph->getVertex('bob')->createEdge($graph->getVertex('carol'));
        $graph->getVertex('alice')->createEdge($graph->getVertex('carol'))->setWeight(2.5);

        return $graph;
    }

    public function test_it_carries_the_nodes_the_edges_and_the_weights(): void
    {
        $labelled = Graphp::import(self::build());

        self::assertSame(4, $labelled->graph->order());
        self::assertSame(3, $labelled->graph->size());
        self::assertFalse($labelled->graph->isDirected());
        self::assertSame(4.5, $labelled->graph->totalWeight());
        self::assertSame(['alice', 'bob', 'carol', 'dave'], $labelled->index->identifiers());
    }

    /**
     * An unweighted edge is reported by graphp as null, and read here as 1.0
     * -- an edge that exists but says nothing about how much.
     */
    public function test_an_unweighted_edge_counts_as_one(): void
    {
        $labelled = Graphp::import(self::build());

        self::assertSame(1.0, $labelled->graph->edgeWeight(
            $labelled->node('alice'),
            $labelled->node('bob'),
        ));
    }

    /** A vertex nobody joined is still a node. */
    public function test_an_isolated_vertex_survives(): void
    {
        $labelled = Graphp::import(self::build());
        $communities = $labelled->communities(Leiden::modularity(seed: 1)->partition($labelled->graph));

        self::assertContains(['dave'], $communities);
    }

    /** A graph with arrows comes through directed, and keeps their direction. */
    public function test_a_directed_graph_keeps_its_arrows(): void
    {
        $graph = new ExternalGraph();
        /** @phpstan-ignore argument.type */
        $graph->createVertex('x');
        /** @phpstan-ignore argument.type */
        $graph->createVertex('y');
        $graph->getVertex('x')->createEdgeTo($graph->getVertex('y'));

        $labelled = Graphp::import($graph);

        self::assertTrue($labelled->graph->isDirected());
        self::assertTrue($labelled->graph->hasEdge($labelled->node('x'), $labelled->node('y')));
        self::assertFalse($labelled->graph->hasEdge($labelled->node('y'), $labelled->node('x')));
    }

    /**
     * graphp allows a graph to mix directed and undirected edges. This library
     * does not, because modularity and betweenness are different formulas in
     * the two cases rather than the same formula with a flag. A mixed graph is
     * read as directed, and each undirected edge becomes a pair of arrows --
     * which is what "undirected" means to a directed algorithm.
     */
    public function test_a_mixed_graph_becomes_directed_with_both_arrows(): void
    {
        $graph = new ExternalGraph();

        foreach (['x', 'y', 'z'] as $id) {
            /** @phpstan-ignore argument.type */
            $graph->createVertex($id);
        }

        $graph->getVertex('x')->createEdgeTo($graph->getVertex('y'));
        $graph->getVertex('y')->createEdge($graph->getVertex('z'));

        $labelled = Graphp::import($graph);

        self::assertTrue($labelled->graph->isDirected());
        self::assertSame(3, $labelled->graph->size(), 'one arrow plus one undirected edge as two');

        self::assertTrue($labelled->graph->hasEdge($labelled->node('y'), $labelled->node('z')));
        self::assertTrue($labelled->graph->hasEdge($labelled->node('z'), $labelled->node('y')));
        self::assertFalse($labelled->graph->hasEdge($labelled->node('y'), $labelled->node('x')));
    }

    public function test_an_empty_graph_imports_to_an_empty_graph(): void
    {
        $labelled = Graphp::import(new ExternalGraph());

        self::assertTrue($labelled->graph->isEmpty());
        self::assertSame([], $labelled->index->identifiers());
    }
}
