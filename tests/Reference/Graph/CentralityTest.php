<?php

declare(strict_types=1);

namespace Vegoia\Tests\Reference\Graph;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Vegoia\Graph\Centrality\Betweenness;
use Vegoia\Graph\Centrality\Closeness;
use Vegoia\Graph\Centrality\PageRank;
use Vegoia\Graph\Graph;
use Vegoia\Tests\Support\GraphFixture;
use Vegoia\Tests\Support\Lre;

/**
 * Centrality against networkx.
 *
 * Each measure has several defensible definitions, and picking a different one
 * silently produces numbers that look reasonable and are not comparable with
 * anyone else's. The conventions are pinned in the fixture generator and
 * restated on each implementation; these tests are what keeps the two in step.
 *
 * @see tools/generate_graph_fixtures.py
 */
#[CoversClass(PageRank::class)]
#[CoversClass(Betweenness::class)]
#[CoversClass(Closeness::class)]
#[Group('reference')]
final class CentralityTest extends TestCase
{
    /** @return iterable<string, array{string}> */
    public static function fixtures(): iterable
    {
        foreach (GraphFixture::names() as $name) {
            yield $name => [$name];
        }
    }

    #[DataProvider('fixtures')]
    public function test_pagerank_agrees_with_networkx(string $name): void
    {
        $fixture = GraphFixture::load($name);
        // Damping is deliberately NOT passed: the fixtures are generated at
        // networkx's 0.85, so leaving it to the default is what pins the
        // default. Passing it explicitly here let a wrong default ship.
        $scores = (new PageRank(tolerance: 1.0e-14))->of($fixture->graph());
        $expected = $fixture->expectedVector('pagerank');

        foreach ($expected as $node => $value) {
            Lre::assertDigits($scores[$node], $value, "{$name}: PageRank of node {$node}", digits: 8);
        }
    }

    #[DataProvider('fixtures')]
    public function test_pagerank_is_a_probability_distribution(string $name): void
    {
        $scores = (new PageRank())->of(GraphFixture::load($name)->graph());

        self::assertEqualsWithDelta(1.0, array_sum($scores), 1.0e-12, "{$name}: scores must sum to 1");

        foreach ($scores as $node => $score) {
            self::assertGreaterThan(0.0, $score, "{$name}: node {$node} has non-positive rank");
        }
    }

    #[DataProvider('fixtures')]
    public function test_betweenness_agrees_with_networkx(string $name): void
    {
        $fixture = GraphFixture::load($name);
        $scores = Betweenness::of($fixture->graph());
        $expected = $fixture->expectedVector('betweenness');

        foreach ($expected as $node => $value) {
            Lre::assertDigits($scores[$node], $value, "{$name}: betweenness of node {$node}", digits: 10);
        }
    }

    #[DataProvider('fixtures')]
    public function test_closeness_agrees_with_networkx(string $name): void
    {
        $fixture = GraphFixture::load($name);
        $scores = Closeness::of($fixture->graph());
        $expected = $fixture->expectedVector('closeness');

        foreach ($expected as $node => $value) {
            Lre::assertDigits($scores[$node], $value, "{$name}: closeness of node {$node}", digits: 10);
        }
    }

    /**
     * A star has a closed form: the hub lies on every shortest path between two
     * leaves, which is (n-1)(n-2)/2 pairs, and no leaf lies on any path at all.
     * Independent of the fixtures, so it catches a generator that drifted.
     */
    public function test_betweenness_of_a_star_matches_the_closed_form(): void
    {
        $graph = Graph::undirected(10, array_map(static fn (int $leaf): array => [0, $leaf], range(1, 9)));
        $scores = Betweenness::of($graph);

        self::assertSame(36.0, $scores[0], 'hub: (n-1)(n-2)/2 with n = 10');

        for ($leaf = 1; $leaf <= 9; $leaf++) {
            self::assertSame(0.0, $scores[$leaf], "leaf {$leaf}");
        }
    }

    public function test_betweenness_of_a_complete_graph_is_zero_everywhere(): void
    {
        $edges = [];
        for ($i = 0; $i < 6; $i++) {
            for ($j = $i + 1; $j < 6; $j++) {
                $edges[] = [$i, $j];
            }
        }

        foreach (Betweenness::of(Graph::undirected(6, $edges)) as $node => $score) {
            self::assertSame(0.0, $score, "node {$node}: every pair is already adjacent");
        }
    }

    /**
     * Damping has a conventional value, and a library that silently used
     * another would produce rankings nobody could reconcile with anyone
     * else's -- while still summing to 1 and still looking plausible.
     */
    public function test_the_default_damping_is_the_conventional_zero_point_eight_five(): void
    {
        $graph = GraphFixture::load('zachary')->graph();

        self::assertSame(
            (new PageRank(damping: 0.85))->of($graph),
            (new PageRank())->of($graph),
        );

        self::assertNotEquals(
            (new PageRank(damping: 0.5))->of($graph),
            (new PageRank())->of($graph),
        );
    }

    public function test_centrality_of_an_empty_graph_is_an_empty_vector(): void
    {
        $empty = Graph::undirected(0);

        self::assertSame([], (new PageRank())->of($empty));
        self::assertSame([], Betweenness::of($empty));
        self::assertSame([], Closeness::of($empty));
    }
}
