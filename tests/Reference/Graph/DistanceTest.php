<?php

declare(strict_types=1);

namespace Vegoia\Tests\Reference\Graph;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Graph\Assortativity;
use Vegoia\Graph\Connectivity;
use Vegoia\Graph\Distance;
use Vegoia\Graph\Graph;
use Vegoia\Graph\Path\BreadthFirst;
use Vegoia\Tests\Support\GraphFixture;
use Vegoia\Tests\Support\Lre;
use Vegoia\Tests\Support\Paths;

/**
 * Distance summaries and degree assortativity, against NetworkX.
 *
 * NetworkX refuses a disconnected graph for diameter, radius and average path
 * length. This library returns INF instead, and the fixture records both
 * readings: the whole-graph one, which is infinite when some pair has no path,
 * and the largest component's, which is what anybody actually looks at. Two of
 * the eleven fixtures are disconnected -- one of them the largest and most
 * realistic -- so a reference that stopped at the exception would have had
 * nothing to say about either.
 *
 * @see tools/generate_structure_fixtures.py
 */
#[CoversClass(Distance::class)]
#[CoversClass(Assortativity::class)]
#[Group('reference')]
final class DistanceTest extends TestCase
{
    /** @return iterable<string, array{string, array<string, mixed>}> */
    public static function graphs(): iterable
    {
        /** @var array{undirected: array<string, array<string, mixed>>} $decoded */
        $decoded = json_decode(
            (string) file_get_contents(Paths::fixture('structure.json')),
            true,
            flags: JSON_THROW_ON_ERROR,
        );

        foreach ($decoded['undirected'] as $name => $entry) {
            yield $name => [$name, $entry];
        }
    }

    /** @param array<string, mixed> $entry */
    #[DataProvider('graphs')]
    public function test_the_eccentricities_match_networkx(string $name, array $entry): void
    {
        /** @var array{distance: array{eccentricity: list<float>, components: int, total_distance: float, reachable_pairs: int}} $entry */
        $expected = $entry['distance'];
        $distance = Distance::of(GraphFixture::load($name)->graph());

        self::assertSame($expected['eccentricity'], $distance->eccentricity, "{$name}: eccentricity");
        self::assertSame($expected['components'], $distance->components, "{$name}: components");
        self::assertSame($expected['total_distance'], $distance->totalDistance, "{$name}: total distance");
        self::assertSame($expected['reachable_pairs'], $distance->reachablePairs, "{$name}: reachable pairs");
    }

    /** @param array<string, mixed> $entry */
    #[DataProvider('graphs')]
    public function test_the_summaries_match_networkx_where_it_will_give_them(
        string $name,
        array $entry,
    ): void {
        /** @var array{distance: array{diameter: float|null, radius: float|null, average_shortest_path_length: float|null}} $entry */
        $expected = $entry['distance'];
        $distance = Distance::of(GraphFixture::load($name)->graph());

        if ($expected['diameter'] === null) {
            // NetworkX raised, which it does exactly when the graph is
            // disconnected. INF is this library's answer, and it is an answer
            // rather than a refusal: the maximum over all pairs of a shortest
            // path length is infinite when some pair has no path, and so is
            // the minimum eccentricity, since every node fails to reach
            // something.
            self::assertFalse($distance->isConnected(), "{$name}: the fixture says disconnected");
            self::assertSame(INF, $distance->diameter(), "{$name}: diameter");
            self::assertSame(INF, $distance->radius(), "{$name}: radius");
            self::assertSame(INF, $distance->averageShortestPathLength(), "{$name}: average path");

            return;
        }

        self::assertTrue($distance->isConnected(), "{$name}: the fixture says connected");
        self::assertSame($expected['diameter'], $distance->diameter(), "{$name}: diameter");
        self::assertSame($expected['radius'], $distance->radius(), "{$name}: radius");

        /** @var float $average */
        $average = $expected['average_shortest_path_length'];
        Lre::assertDigits($distance->averageShortestPathLength(), $average, "{$name}: average path", 13.0);
    }

    /**
     * The largest component's summaries, which are what a caller reads off a
     * disconnected graph and are the reason the eccentricities are defined
     * per component in the first place.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('graphs')]
    public function test_the_largest_component_can_be_read_from_the_public_parts(
        string $name,
        array $entry,
    ): void {
        /** @var array{distance: array{largest_component: array{nodes: int, diameter: float, radius: float}|null}} $entry */
        $largest = $entry['distance']['largest_component'];

        if ($largest === null) {
            self::assertSame(0, Distance::of(GraphFixture::load($name)->graph())->order);

            return;
        }

        $graph = GraphFixture::load($name)->graph();
        $distance = Distance::of($graph);
        $components = Connectivity::components($graph)->communities();

        $biggest = [];

        foreach ($components as $members) {
            if (count($members) > count($biggest)) {
                $biggest = $members;
            }
        }

        self::assertCount($largest['nodes'], $biggest, "{$name}: largest component size");

        $within = [];

        foreach ($biggest as $node) {
            $within[] = $distance->eccentricity[$node];
        }

        /** @var non-empty-list<float> $within */
        self::assertSame($largest['diameter'], max($within), "{$name}: its diameter");
        self::assertSame($largest['radius'], min($within), "{$name}: its radius");
    }

    /**
     * The eccentricity of a node is the farthest anything in its own component
     * gets from it, which is checkable straight from a breadth-first sweep and
     * is a different computation from the one Distance runs.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('graphs')]
    public function test_each_eccentricity_is_the_farthest_reachable_node(
        string $name,
        array $entry,
    ): void {
        $graph = GraphFixture::load($name)->graph();
        $distance = Distance::of($graph);

        for ($node = 0; $node < $graph->order(); $node++) {
            $farthest = 0.0;

            foreach (BreadthFirst::distancesFrom($graph, $node) as $hops) {
                // -1 marks unreachable, and an unreachable node is in another
                // component and so says nothing about this one's eccentricity.
                if ($hops > $farthest) {
                    $farthest = $hops;
                }
            }

            self::assertSame(
                $farthest,
                $distance->eccentricity[$node],
                "{$name}: eccentricity of node {$node}",
            );
        }
    }

    /** @param array<string, mixed> $entry */
    #[DataProvider('graphs')]
    public function test_the_degree_assortativity_matches_networkx(string $name, array $entry): void
    {
        /** @var array{degree_assortativity: float|null} $entry */
        $graph = GraphFixture::load($name)->graph();

        if ($entry['degree_assortativity'] === null) {
            // Every edge joins two nodes of the same degree, so the
            // correlation has no denominator. NetworkX returns nan; this
            // refuses, which is what Correlation::pearson already does for a
            // sample with no variation, and this is literally that.
            $this->expectException(InvalidArgument::class);
            $this->expectExceptionMessageMatches('/no variation|regular/i');

            Assortativity::degree($graph);

            return;
        }

        Lre::assertDigits(
            Assortativity::degree($graph),
            $entry['degree_assortativity'],
            "{$name}: degree assortativity",
            12.0,
        );
    }

    /**
     * A self-loop contributes no pair of endpoints.
     *
     * Not covered by any fixture -- none of the eleven has one -- and found by
     * deliberately breaking the rule and watching nothing fail. A node joined
     * to itself says nothing about who it prefers to attach to, and counting
     * it would force a (d, d) pair that drags the correlation mechanically
     * towards +1.
     *
     * All three readings differ and NetworkX agrees with neither of the
     * obvious ones, so this is pinned by hand against a direct Pearson
     * computation rather than against it: on a four-cycle with one self-loop,
     * excluding the loop gives -1/3, including it once gives +1/6, and
     * NetworkX reports 0.0.
     */
    public function test_a_self_loop_contributes_no_pair_of_endpoints(): void
    {
        $withLoop = Graph::undirected(4, [[0, 1], [1, 2], [2, 3], [3, 0], [0, 0]]);

        // The loop still counts twice towards the degree, as it does
        // everywhere else in this library -- node 0 has degree 4 here.
        self::assertSame(4, $withLoop->degree(0));

        self::assertEqualsWithDelta(-1.0 / 3.0, Assortativity::degree($withLoop), 1.0e-14);
        self::assertNotEqualsWithDelta(1.0 / 6.0, Assortativity::degree($withLoop), 1.0e-6);
    }

    public function test_an_empty_graph_is_answerable(): void
    {
        $distance = Distance::of(Graph::undirected(0));

        self::assertSame([], $distance->eccentricity);
        self::assertSame(0, $distance->components);
        self::assertSame(0.0, $distance->diameter());
        self::assertSame(0.0, $distance->radius());
        self::assertSame(0.0, $distance->averageShortestPathLength());
        self::assertTrue($distance->isConnected());
    }

    public function test_a_single_node_has_no_distance_to_anywhere(): void
    {
        $distance = Distance::of(Graph::undirected(1));

        self::assertSame([0.0], $distance->eccentricity);
        self::assertSame(1, $distance->components);
        self::assertSame(0, $distance->reachablePairs);
        self::assertSame(0.0, $distance->averageShortestPathLength());
    }

    public function test_directed_graphs_are_refused(): void
    {
        $this->expectException(InvalidArgument::class);
        $this->expectExceptionMessageMatches('/undirected graphs only/');

        Distance::of(Graph::directed(3, [[0, 1], [1, 2]]));
    }

    public function test_assortativity_refuses_a_directed_graph(): void
    {
        $this->expectException(InvalidArgument::class);
        $this->expectExceptionMessageMatches('/undirected graphs only/');

        Assortativity::degree(Graph::directed(3, [[0, 1], [1, 2]]));
    }

    public function test_assortativity_refuses_a_graph_with_no_edges(): void
    {
        $this->expectException(InvalidArgument::class);

        Assortativity::degree(Graph::undirected(3));
    }
}
