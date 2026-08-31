<?php

declare(strict_types=1);

namespace Vegoia\Tests\Reference\Graph;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Vegoia\Exception\InvalidArgument;
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

    #[DataProvider('fixtures')]
    public function test_personalised_pagerank_agrees_with_networkx(string $name): void
    {
        $fixture = GraphFixture::load($name);

        $seeds = [];
        foreach ($fixture->expectedVector('personalisation_nodes') as $node) {
            $seeds[(int) $node] = 1.0;
        }

        $scores = (new PageRank(tolerance: 1.0e-14))->of($fixture->graph(), $seeds);

        foreach ($fixture->expectedVector('pagerank_personalised') as $node => $value) {
            Lre::assertDigits($scores[$node], $value, "{$name}: personalised rank of {$node}", digits: 8);
        }
    }

    #[DataProvider('fixtures')]
    public function test_weighted_betweenness_agrees_with_networkx(string $name): void
    {
        $fixture = GraphFixture::load($name);
        $scores = Betweenness::weighted($fixture->graph());

        foreach ($fixture->expectedVector('betweenness_weighted') as $node => $value) {
            Lre::assertDigits($scores[$node], $value, "{$name}: weighted betweenness of {$node}", digits: 10);
        }
    }

    /**
     * Aiming the teleportation is the point: the seeds must gain and the rest
     * must lose, or the personalisation is being ignored.
     */
    public function test_personalisation_concentrates_rank_on_its_seeds(): void
    {
        $graph = GraphFixture::load('zachary')->graph();
        $rank = new PageRank();

        $plain = $rank->of($graph);
        $biased = $rank->of($graph, [0 => 1.0]);

        self::assertGreaterThan($plain[0], $biased[0], 'the seed must gain');
        self::assertEqualsWithDelta(1.0, array_sum($biased), 1.0e-12, 'still a distribution');

        // The mass has to come from somewhere.
        $lost = 0;
        for ($node = 1; $node < $graph->order(); $node++) {
            if ($biased[$node] < $plain[$node]) {
                $lost++;
            }
        }

        self::assertGreaterThan($graph->order() / 2, $lost, 'most other nodes must lose rank');
    }

    /** A uniform personalisation is the ordinary walk, by definition. */
    public function test_uniform_personalisation_reproduces_plain_pagerank(): void
    {
        $graph = GraphFixture::load('zachary')->graph();
        $uniform = array_fill(0, $graph->order(), 1.0);

        $plain = (new PageRank(tolerance: 1.0e-14))->of($graph);
        $spread = (new PageRank(tolerance: 1.0e-14))->of($graph, $uniform);

        foreach ($plain as $node => $value) {
            self::assertEqualsWithDelta($value, $spread[$node], 1.0e-12, "node {$node}");
        }
    }

    public function test_personalisation_is_validated(): void
    {
        $graph = GraphFixture::load('zachary')->graph();

        $this->expectException(InvalidArgument::class);

        (new PageRank())->of($graph, [999 => 1.0]);
    }

    public function test_a_personalisation_with_no_weight_is_refused(): void
    {
        $graph = GraphFixture::load('zachary')->graph();

        $this->expectException(InvalidArgument::class);

        (new PageRank())->of($graph, [0 => 0.0, 1 => 0.0]);
    }

    /**
     * Weights are distances here, not capacities: the direct heavy edge is a
     * longer route than two light ones, so the middle node carries the traffic.
     */
    public function test_weighted_betweenness_treats_weight_as_distance(): void
    {
        $graph = Graph::undirected(3, [[0, 1, 1.0], [1, 2, 1.0], [0, 2, 10.0]]);

        self::assertSame(0.0, Betweenness::of($graph)[1], 'by hops, 0-2 is direct');
        self::assertSame(1.0, Betweenness::weighted($graph)[1], 'by weight, everything goes through 1');
    }

    public function test_weighted_betweenness_refuses_negative_weights(): void
    {
        $this->expectException(InvalidArgument::class);

        Betweenness::weighted(Graph::undirected(3, [[0, 1, 1.0], [1, 2, -1.0]]));
    }

    public function test_centrality_of_an_empty_graph_is_an_empty_vector(): void
    {
        $empty = Graph::undirected(0);

        self::assertSame([], (new PageRank())->of($empty));
        self::assertSame([], Betweenness::of($empty));
        self::assertSame([], Betweenness::weighted($empty));
        self::assertSame([], Closeness::of($empty));
    }
}
