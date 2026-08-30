<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit\Graph;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;
use Vegoia\Graph\Connectivity;
use Vegoia\Graph\Graph;
use Vegoia\Tests\Support\GraphFixture;

#[CoversClass(Connectivity::class)]
final class ConnectivityTest extends TestCase
{
    public function test_a_path_is_connected(): void
    {
        $graph = Graph::undirected(4, [[0, 1], [1, 2], [2, 3]]);

        self::assertTrue(Connectivity::isConnected($graph));
        self::assertSame(1, Connectivity::components($graph)->count());
    }

    public function test_it_finds_separate_components(): void
    {
        $graph = Graph::undirected(5, [[0, 1], [2, 3]]);

        self::assertFalse(Connectivity::isConnected($graph));

        $components = Connectivity::components($graph);

        self::assertSame(3, $components->count(), 'two pairs and one isolated node');
        self::assertSame(
            $components->communityOf(0),
            $components->communityOf(1),
        );
        self::assertNotSame(
            $components->communityOf(0),
            $components->communityOf(2),
        );
    }

    public function test_an_empty_graph_is_connected_by_convention(): void
    {
        self::assertTrue(Connectivity::isConnected(Graph::undirected(0)));
        self::assertSame(0, Connectivity::components(Graph::undirected(0))->count());
    }

    public function test_it_decides_whether_a_subset_induces_a_connected_subgraph(): void
    {
        // 0-1-2 and 3 attached only to 1: {0,2,3} is disconnected without 1.
        $graph = Graph::undirected(4, [[0, 1], [1, 2], [1, 3]]);

        self::assertTrue(Connectivity::inducesConnectedSubgraph($graph, [0, 1, 2]));
        self::assertFalse(Connectivity::inducesConnectedSubgraph($graph, [0, 2, 3]));
        self::assertTrue(Connectivity::inducesConnectedSubgraph($graph, [2]));
        self::assertTrue(Connectivity::inducesConnectedSubgraph($graph, []));
    }

    /** @return iterable<string, array{string}> */
    public static function fixtures(): iterable
    {
        foreach (GraphFixture::names() as $name) {
            yield $name => [$name];
        }
    }

    #[DataProvider('fixtures')]
    public function test_component_count_agrees_with_networkx(string $name): void
    {
        $fixture = GraphFixture::load($name);

        /** @var array{components: int} $expected */
        $expected = $fixture->expected;

        self::assertSame(
            $expected['components'],
            Connectivity::components($fixture->graph())->count(),
            "{$name}: number of connected components",
        );
    }
}
