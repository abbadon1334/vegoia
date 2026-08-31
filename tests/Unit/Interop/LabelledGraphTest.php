<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit\Interop;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Graph\Centrality\PageRank;
use Vegoia\Graph\Community\Leiden;
use Vegoia\Graph\Graph;
use Vegoia\Graph\NodeIndex;
use Vegoia\Interop\LabelledGraph;

/**
 * Getting a graph in, and getting the names back out.
 *
 * The three shapes data arrives in -- edge lists, adjacency maps, adjacency
 * matrices -- must produce the same graph when they describe the same one.
 * That equivalence is most of this file, because it is the property a caller
 * relies on without ever stating it: whichever way they happened to have the
 * data, the answer should not depend on it.
 */
#[CoversClass(LabelledGraph::class)]
final class LabelledGraphTest extends TestCase
{
    /**
     * The same triangle plus an isolated node, written four ways.
     *
     * The provider hands back a closure rather than a built graph, and that is
     * not a style choice: code run inside a data provider is not attributed to
     * the test that consumes it, so building here left the constructors
     * looking untested. Mutation testing said so -- flipping the `$directed`
     * default to true survived every assertion below.
     *
     * @return iterable<string, array{callable(): LabelledGraph}>
     */
    public static function theSameGraph(): iterable
    {
        yield 'edge list' => [
            static fn (): LabelledGraph => LabelledGraph::fromEdges(
                [['a', 'b'], ['b', 'c'], ['a', 'c']],
                ['a', 'b', 'c', 'd'],
            ),
        ];

        yield 'adjacency list' => [
            static fn (): LabelledGraph => LabelledGraph::fromAdjacency(
                ['a' => ['b', 'c'], 'b' => ['c'], 'c' => [], 'd' => []],
            ),
        ];

        yield 'adjacency map with weights' => [
            static fn (): LabelledGraph => LabelledGraph::fromAdjacency([
                'a' => ['b' => 1.0, 'c' => 1.0],
                'b' => ['c' => 1.0],
                'c' => [],
                'd' => [],
            ]),
        ];

        yield 'matrix' => [
            static fn (): LabelledGraph => LabelledGraph::fromMatrix(
                [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
                ['a', 'b', 'c', 'd'],
            ),
        ];
    }

    /** @param callable(): LabelledGraph $build */
    #[DataProvider('theSameGraph')]
    public function test_every_input_shape_describes_the_same_graph(callable $build): void
    {
        $labelled = $build();

        self::assertSame(4, $labelled->graph->order());
        self::assertSame(3, $labelled->graph->size());
        self::assertSame(3.0, $labelled->graph->totalWeight());
        self::assertFalse($labelled->graph->isDirected());

        self::assertSame(['a', 'b', 'c', 'd'], $labelled->index->identifiers());
        self::assertSame(0, $labelled->node('a'));
        self::assertSame('d', $labelled->identifier(3));

        self::assertSame(2, $labelled->graph->degree($labelled->node('a')));
        self::assertSame(0, $labelled->graph->degree($labelled->node('d')));
    }

    /**
     * An entity nobody linked is not a degenerate case; it is exactly what a
     * knowledge graph should be able to report on.
     */
    /** @param callable(): LabelledGraph $build */
    #[DataProvider('theSameGraph')]
    public function test_an_isolated_node_survives_the_round_trip(callable $build): void
    {
        $labelled = $build();
        $communities = $labelled->communities(Leiden::modularity(seed: 1)->partition($labelled->graph));

        self::assertContains(['d'], $communities, 'the isolated node lost its own community');
    }

    /** Weights given per edge reach the graph as given. */
    public function test_an_edge_list_carries_its_weights(): void
    {
        $labelled = LabelledGraph::fromEdges([
            ['claim-a', 'claim-b', 0.9],
            ['claim-b', 'claim-c', 0.4],
        ]);

        self::assertSame(3, $labelled->graph->order());
        self::assertSame(2, $labelled->graph->size());
        self::assertSame(
            0.9,
            $labelled->graph->edgeWeight($labelled->node('claim-a'), $labelled->node('claim-b')),
        );
    }

    /**
     * Nodes with no edges would otherwise vanish, and an isolated claim is a
     * result: it belongs to a community of its own, not to nothing.
     */
    public function test_isolated_nodes_survive_when_declared_up_front(): void
    {
        $labelled = LabelledGraph::fromEdges([['a', 'b']], ['a', 'b', 'orphan']);

        self::assertSame(3, $labelled->graph->order());
        self::assertSame(0, $labelled->graph->degree($labelled->node('orphan')));
    }

    /** Weights come through as weights, not as counts of edges. */
    public function test_weights_are_carried_rather_than_counted(): void
    {
        $labelled = LabelledGraph::fromAdjacency(['a' => ['b' => 2.5, 'c' => 0.5]]);

        self::assertSame(3.0, $labelled->graph->totalWeight());
        self::assertSame(2.5, $labelled->graph->edgeWeight($labelled->node('a'), $labelled->node('b')));
        self::assertTrue($labelled->graph->isWeighted());
    }

    /**
     * An undirected matrix states each edge twice, and the graph sums parallel
     * edges, so letting both halves through would double every weight.
     */
    public function test_an_undirected_matrix_is_not_counted_twice(): void
    {
        $labelled = LabelledGraph::fromMatrix([[0, 2.5], [2.5, 0]]);

        self::assertSame(1, $labelled->graph->size());
        self::assertSame(2.5, $labelled->graph->totalWeight());
    }

    /**
     * Without labels the rows are named by their position, so identifiers
     * still round-trip and the result behaves like every other one here.
     */
    public function test_an_unlabelled_matrix_names_its_rows_by_position(): void
    {
        $labelled = LabelledGraph::fromMatrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]]);

        self::assertSame(['0', '1', '2'], $labelled->index->identifiers());
        self::assertSame(1, $labelled->node('1'));
    }

    /**
     * The diagonal is kept, so a self-loop written in a matrix survives.
     *
     * It is the reason the undirected case takes the upper triangle including
     * the diagonal rather than strictly above it -- skipping the diagonal
     * would silently drop every self-loop, and a self-loop is a real thing to
     * say about an entity.
     */
    public function test_a_self_loop_on_the_diagonal_survives(): void
    {
        $labelled = LabelledGraph::fromMatrix([[3.0, 1.0], [1.0, 0.0]]);

        self::assertSame(3.0, $labelled->graph->selfLoopWeight(0));
        self::assertSame(4.0, $labelled->graph->totalWeight());
    }

    /** A directed matrix is not symmetric and both halves are edges. */
    public function test_a_directed_matrix_keeps_both_halves(): void
    {
        $labelled = LabelledGraph::fromMatrix([[0, 1], [2, 0]], directed: true);

        self::assertTrue($labelled->graph->isDirected());
        self::assertSame(2, $labelled->graph->size());
        self::assertSame(1.0, $labelled->graph->edgeWeight(0, 1));
        self::assertSame(2.0, $labelled->graph->edgeWeight(1, 0));
    }

    /** Numbering follows first appearance, so the same input numbers the same. */
    public function test_the_numbering_is_reproducible(): void
    {
        $edges = [['zebra', 'apple'], ['apple', 'mango']];

        $first = LabelledGraph::fromEdges($edges);
        $second = LabelledGraph::fromEdges($edges);

        self::assertSame(['zebra', 'apple', 'mango'], $first->index->identifiers());
        self::assertSame($first->index->identifiers(), $second->index->identifiers());
    }

    /** Integer names are names, not positions. */
    public function test_numeric_identifiers_are_treated_as_names(): void
    {
        $labelled = LabelledGraph::fromEdges([[100, 200], [200, 300]]);

        self::assertSame(3, $labelled->graph->order());
        self::assertSame(['100', '200', '300'], $labelled->index->identifiers());
        self::assertSame(0, $labelled->node(100));
    }

    /** Per-node results come back keyed by name. */
    public function test_results_can_be_read_back_under_their_names(): void
    {
        $labelled = LabelledGraph::fromEdges([['a', 'b'], ['b', 'c']]);
        $named = $labelled->named(new PageRank()->of($labelled->graph));

        self::assertSame(['a', 'b', 'c'], array_keys($named));
        self::assertGreaterThan($named['a'], $named['b'], 'the middle of a path ranks highest');
    }

    public function test_membership_and_communities_describe_the_same_partition(): void
    {
        $labelled = LabelledGraph::fromEdges([['a', 'b'], ['c', 'd']]);
        $partition = Leiden::modularity(seed: 1)->partition($labelled->graph);

        $membership = $labelled->membership($partition);
        $communities = $labelled->communities($partition);

        foreach ($communities as $index => $names) {
            foreach ($names as $name) {
                self::assertSame($index, $membership[$name], "{$name} is in two places at once");
            }
        }
    }

    /** @return iterable<string, array{callable(): mixed, string}> */
    public static function malformedInput(): iterable
    {
        yield 'an edge that is not a pair' => [
            /** @phpstan-ignore argument.type */
            static fn () => LabelledGraph::fromEdges([['a']]),
            'expected [from, to]',
        ];
        yield 'a ragged matrix' => [
            static fn () => LabelledGraph::fromMatrix([[0, 1], [1]]),
            'Row 1 has 1 entries',
        ];
        yield 'the wrong number of labels' => [
            static fn () => LabelledGraph::fromMatrix([[0, 1], [1, 0]], ['only one']),
            '2 rows and 1 labels',
        ];
        yield 'a neighbour that is neither a name nor a number' => [
            /** @phpstan-ignore argument.type */
            static fn () => LabelledGraph::fromAdjacency(['a' => [['nested']]]),
            'neither a name nor a number',
        ];
        yield 'an index that does not fit the graph' => [
            static fn () => new LabelledGraph(Graph::undirected(3), NodeIndex::of(['a'])),
            '3 nodes and the index names 1',
        ];
    }

    /** @param callable(): mixed $call */
    #[DataProvider('malformedInput')]
    public function test_it_refuses_input_it_cannot_read(callable $call, string $identifies): void
    {
        $this->expectException(InvalidArgument::class);
        $this->expectExceptionMessageMatches('/' . preg_quote($identifies, '/') . '/i');

        $call();
    }

    public function test_an_unknown_name_is_refused(): void
    {
        $labelled = LabelledGraph::fromEdges([['a', 'b']]);

        $this->expectException(InvalidArgument::class);

        $labelled->node('nobody');
    }
}
