<?php

declare(strict_types=1);

namespace Vegoia\Tests\Reference\Graph;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Graph\Graph;
use Vegoia\Graph\Path\BreadthFirst;
use Vegoia\Graph\Path\Dijkstra;
use Vegoia\Tests\Support\GraphFixture;

#[CoversClass(BreadthFirst::class)]
#[CoversClass(Dijkstra::class)]
#[Group('reference')]
final class PathTest extends TestCase
{
    /** @return iterable<string, array{string}> */
    public static function fixtures(): iterable
    {
        foreach (GraphFixture::names() as $name) {
            yield $name => [$name];
        }
    }

    #[DataProvider('fixtures')]
    public function test_hop_distances_agree_with_networkx(string $name): void
    {
        $fixture = GraphFixture::load($name);

        /** @var array{paths: array{bfs_from_0: list<float>}} $expected */
        $expected = $fixture->expected;

        self::assertSame(
            $expected['paths']['bfs_from_0'],
            BreadthFirst::distancesFrom($fixture->graph(), 0),
            "{$name}: hop distances from node 0 (-1 where unreachable)",
        );
    }

    #[DataProvider('fixtures')]
    public function test_weighted_distances_agree_with_networkx(string $name): void
    {
        $fixture = GraphFixture::load($name);

        /** @var array{paths: array{dijkstra_from_0: list<float>}} $expected */
        $expected = $fixture->expected;

        $computed = Dijkstra::distancesFrom($fixture->graph(), 0);

        foreach ($expected['paths']['dijkstra_from_0'] as $node => $distance) {
            self::assertEqualsWithDelta(
                $distance,
                $computed[$node],
                1.0e-12,
                "{$name}: weighted distance to node {$node}",
            );
        }
    }

    public function test_a_path_graph_has_the_distances_you_would_count_on_your_fingers(): void
    {
        $graph = Graph::undirected(4, [[0, 1], [1, 2], [2, 3]]);

        self::assertSame([0.0, 1.0, 2.0, 3.0], BreadthFirst::distancesFrom($graph, 0));
        self::assertSame([2.0, 1.0, 0.0, 1.0], BreadthFirst::distancesFrom($graph, 2));
    }

    public function test_unreachable_nodes_are_reported_as_minus_one(): void
    {
        $graph = Graph::undirected(4, [[0, 1], [2, 3]]);

        self::assertSame([0.0, 1.0, -1.0, -1.0], BreadthFirst::distancesFrom($graph, 0));
        self::assertSame([0.0, 1.0, -1.0, -1.0], Dijkstra::distancesFrom($graph, 0));
    }

    /**
     * The cheap route is not the short one: 0-1-2 costs 2 hops but weight 2,
     * while the direct edge 0-2 is 1 hop and weight 10.
     */
    public function test_dijkstra_prefers_weight_where_breadth_first_prefers_hops(): void
    {
        $graph = Graph::undirected(3, [[0, 1, 1.0], [1, 2, 1.0], [0, 2, 10.0]]);

        self::assertSame([0.0, 1.0, 1.0], BreadthFirst::distancesFrom($graph, 0));
        self::assertSame([0.0, 1.0, 2.0], Dijkstra::distancesFrom($graph, 0));
    }

    public function test_dijkstra_refuses_negative_weights_rather_than_returning_nonsense(): void
    {
        $graph = Graph::undirected(3, [[0, 1, 1.0], [1, 2, -5.0]]);

        $this->expectException(InvalidArgument::class);

        Dijkstra::distancesFrom($graph, 0);
    }

    public function test_it_reconstructs_the_path_and_not_only_its_length(): void
    {
        $graph = Graph::undirected(5, [[0, 1, 1.0], [1, 4, 1.0], [0, 2, 1.0], [2, 3, 1.0], [3, 4, 1.0]]);

        self::assertSame([0, 1, 4], Dijkstra::shortestPath($graph, 0, 4));
        self::assertSame([], Dijkstra::shortestPath(Graph::undirected(2), 0, 1), 'no path');
        self::assertSame([2], Dijkstra::shortestPath($graph, 2, 2), 'a node reaches itself');
    }
}
