<?php

declare(strict_types=1);

namespace Vegoia\Tests\Reference\Graph;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Graph\Connectivity;
use Vegoia\Graph\Graph;
use Vegoia\Graph\SpanningTree;
use Vegoia\Tests\Support\GraphFixture;
use Vegoia\Tests\Support\Lre;
use Vegoia\Tests\Support\Paths;

/**
 * Minimum and maximum spanning forests, against NetworkX.
 *
 * The total weight and the edge count are compared, not the edge set. A
 * spanning tree is not unique when weights tie, and most of these fixtures are
 * unweighted, so every edge ties with every other: two correct
 * implementations disagree about which edges they picked and agree exactly
 * about what the result costs. Asserting the edges would be asserting
 * NetworkX's tie-breaking.
 *
 * @see tools/generate_spanning_tree_fixtures.py
 */
#[CoversClass(SpanningTree::class)]
#[Group('reference')]
final class SpanningTreeTest extends TestCase
{
    /** @return iterable<string, array{string, array<string, mixed>}> */
    public static function graphs(): iterable
    {
        /** @var array{graphs: array<string, array<string, mixed>>} $decoded */
        $decoded = json_decode(
            (string) file_get_contents(Paths::fixture('spanning_tree.json')),
            true,
            flags: JSON_THROW_ON_ERROR,
        );

        foreach ($decoded['graphs'] as $name => $entry) {
            yield $name => [$name, $entry];
        }
    }

    /** @param array<string, mixed> $entry */
    #[DataProvider('graphs')]
    public function test_the_forests_cost_what_networkx_says(string $name, array $entry): void
    {
        /** @var array{edges_expected: int, minimum: array{edges: int, weight: float}, maximum: array{edges: int, weight: float}} $entry */
        $graph = GraphFixture::load($name)->graph();

        foreach (['minimum' => false, 'maximum' => true] as $which => $dearest) {
            $edges = $dearest ? SpanningTree::maximum($graph) : SpanningTree::minimum($graph);

            self::assertCount($entry['edges_expected'], $edges, "{$name}: {$which} edge count");

            $weight = 0.0;

            foreach ($edges as [, , $edgeWeight]) {
                $weight += $edgeWeight;
            }

            /** @var array{edges: int, weight: float} $expected */
            $expected = $entry[$which];

            Lre::assertDigits($weight, $expected['weight'], "{$name}: {$which} weight", 13.0);
            Lre::assertDigits(
                SpanningTree::weight($graph, $dearest),
                $expected['weight'],
                "{$name}: {$which} weight via weight()",
                13.0,
            );
        }
    }

    /**
     * A forest of n nodes over c components has exactly n - c edges, spans
     * every node, and contains no cycle.
     *
     * The definition, checked directly rather than through the fixture. It
     * would catch an implementation that agreed on the total weight by
     * accident -- taking one edge too many and one too few, say.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('graphs')]
    public function test_the_result_is_a_spanning_forest(string $name, array $entry): void
    {
        $graph = GraphFixture::load($name)->graph();
        $components = Connectivity::components($graph)->count();

        foreach ([SpanningTree::minimum($graph), SpanningTree::maximum($graph)] as $edges) {
            $forest = SpanningTree::asGraph($graph, $edges);

            self::assertSame(
                $graph->order() - $components,
                count($edges),
                "{$name}: a forest over {$components} components must have n - c edges",
            );

            self::assertSame(
                $components,
                Connectivity::components($forest)->count(),
                "{$name}: the forest must join exactly what the graph joined",
            );

            // Acyclicity follows from the count above by itself, and is
            // checked separately and properly in
            // test_every_edge_of_the_forest_is_load_bearing.
        }
    }

    /**
     * Every edge of a tree is a bridge of it, since removing any one splits
     * the tree in two. A forest with a non-bridge edge is not a forest.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('graphs')]
    public function test_every_edge_of_the_forest_is_load_bearing(string $name, array $entry): void
    {
        $graph = GraphFixture::load($name)->graph();
        $edges = SpanningTree::minimum($graph);
        $forest = SpanningTree::asGraph($graph, $edges);

        self::assertCount(
            count($edges),
            Connectivity::bridges($forest),
            "{$name}: a forest in which some edge is not a bridge contains a cycle",
        );
    }

    /**
     * The maximum forest costs at least what the minimum does, and on a
     * weighted graph strictly more.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('graphs')]
    public function test_the_maximum_is_never_cheaper_than_the_minimum(string $name, array $entry): void
    {
        $graph = GraphFixture::load($name)->graph();

        $minimum = SpanningTree::weight($graph, dearest: false);
        $maximum = SpanningTree::weight($graph, dearest: true);

        self::assertGreaterThanOrEqual($minimum, $maximum, $name);

        if ($graph->isWeighted()) {
            self::assertGreaterThan(
                $minimum,
                $maximum,
                "{$name}: the two agree on a weighted graph, so the sort is not being reversed",
            );
        }
    }

    /**
     * The minimum forest is minimal: no other spanning forest is cheaper.
     *
     * Checked by the exchange property rather than by enumerating trees. For
     * every edge not in the forest, adding it makes exactly one cycle, and
     * that edge must be no lighter than the heaviest edge already on the path
     * between its endpoints -- otherwise swapping the two would give a
     * cheaper forest, and the answer was not minimal.
     */
    public function test_no_excluded_edge_would_improve_the_minimum(): void
    {
        $graph = GraphFixture::load('lesmis')->graph();
        $edges = SpanningTree::minimum($graph);
        $forest = SpanningTree::asGraph($graph, $edges);

        $inForest = [];

        foreach ($edges as [$from, $to]) {
            $inForest["{$from}-{$to}"] = true;
        }

        foreach ($graph->edges() as [$from, $to, $weight]) {
            if (isset($inForest["{$from}-{$to}"]) || $from === $to) {
                continue;
            }

            $heaviest = self::heaviestOnPath($forest, $from, $to);

            self::assertGreaterThanOrEqual(
                $heaviest,
                $weight,
                sprintf(
                    'edge %d-%d weighs %.4f and the path between its endpoints carries an edge '
                    . 'of %.4f, so swapping them would give a cheaper forest',
                    $from,
                    $to,
                    $weight,
                    $heaviest,
                ),
            );
        }
    }

    /** The heaviest edge on the unique path between two nodes of a forest. */
    private static function heaviestOnPath(Graph $forest, int $from, int $to): float
    {
        /** @var list<array{int, int, float}> $stack */
        $stack = [[$from, -1, -INF]];
        $seen = [$from => true];

        while ($stack !== []) {
            /** @var array{int, int, float} $frame */
            $frame = array_pop($stack);
            [$node, $parent, $heaviest] = $frame;

            if ($node === $to) {
                return $heaviest;
            }

            foreach ($forest->neighbours($node) as $neighbour => $weight) {
                if ($neighbour === $parent || isset($seen[$neighbour])) {
                    continue;
                }

                $seen[$neighbour] = true;
                $stack[] = [$neighbour, $node, max($heaviest, $weight)];
            }
        }

        return -INF;
    }

    public function test_a_directed_graph_is_refused(): void
    {
        $this->expectException(InvalidArgument::class);
        $this->expectExceptionMessageMatches('/undirected graphs only/');

        SpanningTree::minimum(Graph::directed(3, [[0, 1], [1, 2]]));
    }

    public function test_an_empty_graph_has_an_empty_forest(): void
    {
        self::assertSame([], SpanningTree::minimum(Graph::undirected(0)));
        self::assertSame(0.0, SpanningTree::weight(Graph::undirected(0)));
    }

    /** A self-loop can never join two pieces, so it is never in the forest. */
    public function test_self_loops_are_never_chosen(): void
    {
        $graph = Graph::undirected(3, [[0, 0, 0.1], [0, 1, 5.0], [1, 2, 5.0]]);

        self::assertCount(2, SpanningTree::minimum($graph));
        self::assertSame(10.0, SpanningTree::weight($graph));
    }
}
