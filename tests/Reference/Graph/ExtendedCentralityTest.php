<?php

declare(strict_types=1);

namespace Vegoia\Tests\Reference\Graph;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Graph\Centrality\Eigenvector;
use Vegoia\Graph\Centrality\Harmonic;
use Vegoia\Graph\Centrality\Katz;
use Vegoia\Graph\Clustering;
use Vegoia\Graph\Graph;
use Vegoia\Graph\KCore;
use Vegoia\Tests\Support\GraphFixture;
use Vegoia\Tests\Support\Lre;

/**
 * The measures added to close the gap against networkx.
 *
 * Two of them exposed real problems that only appear on particular graph
 * shapes, which is why the fixtures include those shapes deliberately:
 *
 *   * eigenvector centrality by plain power iteration never converges on a
 *     bipartite graph -- the spectrum is symmetric, so the two largest
 *     eigenvalues match in magnitude and the vector oscillates. star_10 and
 *     davis are bipartite and caught it.
 *
 *   * Katz diverges unless alpha is below 1/lambda_max, and on the weighted
 *     fixtures the conventional 0.05 already is not. It returned NaN. The
 *     alpha each fixture was generated with is recorded alongside its values.
 */
#[CoversClass(Eigenvector::class)]
#[CoversClass(Harmonic::class)]
#[CoversClass(Katz::class)]
#[CoversClass(Clustering::class)]
#[CoversClass(KCore::class)]
#[Group('reference')]
final class ExtendedCentralityTest extends TestCase
{
    /** @return iterable<string, array{string}> */
    public static function fixtures(): iterable
    {
        foreach (GraphFixture::names() as $name) {
            yield $name => [$name];
        }
    }

    #[DataProvider('fixtures')]
    public function test_eigenvector_centrality_agrees_with_networkx(string $name): void
    {
        $fixture = GraphFixture::load($name);
        $scores = (new Eigenvector())->of($fixture->graph());

        foreach ($fixture->expectedVector('eigenvector') as $node => $expected) {
            self::assertEqualsWithDelta($expected, $scores[$node], 1.0e-9, "{$name}: node {$node}");
        }
    }

    #[DataProvider('fixtures')]
    public function test_eigenvector_centrality_is_a_unit_vector(string $name): void
    {
        $norm = 0.0;

        foreach ((new Eigenvector())->of(GraphFixture::load($name)->graph()) as $score) {
            self::assertGreaterThanOrEqual(0.0, $score, "{$name}: negative score");
            $norm += $score * $score;
        }

        self::assertEqualsWithDelta(1.0, $norm, 1.0e-9, "{$name}: not L2-normalised");
    }

    #[DataProvider('fixtures')]
    public function test_harmonic_centrality_agrees_with_networkx(string $name): void
    {
        $fixture = GraphFixture::load($name);
        $scores = Harmonic::of($fixture->graph());

        foreach ($fixture->expectedVector('harmonic') as $node => $expected) {
            Lre::assertDigits($scores[$node], $expected, "{$name}: harmonic of node {$node}", digits: 12);
        }
    }

    #[DataProvider('fixtures')]
    public function test_katz_centrality_agrees_with_networkx(string $name): void
    {
        $fixture = GraphFixture::load($name);
        $alpha = $fixture->expectedVector('katz_alpha')[0];
        $scores = (new Katz($alpha))->of($fixture->graph());

        foreach ($fixture->expectedVector('katz') as $node => $expected) {
            self::assertEqualsWithDelta($expected, $scores[$node], 1.0e-9, "{$name}: node {$node}");
        }
    }

    #[DataProvider('fixtures')]
    public function test_clustering_coefficients_agree_with_networkx(string $name): void
    {
        $fixture = GraphFixture::load($name);
        $graph = $fixture->graph();
        $coefficients = Clustering::coefficients($graph);
        $triangles = Clustering::triangles($graph);

        foreach ($fixture->expectedVector('clustering') as $node => $expected) {
            self::assertEqualsWithDelta($expected, $coefficients[$node], 1.0e-12, "{$name}: clustering of {$node}");
        }

        foreach ($fixture->expectedVector('triangles') as $node => $expected) {
            self::assertSame((int) $expected, $triangles[$node], "{$name}: triangles through {$node}");
        }
    }

    #[DataProvider('fixtures')]
    public function test_the_two_clustering_summaries_agree_with_networkx(string $name): void
    {
        $fixture = GraphFixture::load($name);
        $graph = $fixture->graph();

        /** @var array{transitivity: float, average_clustering: float, triangle_total: int} $expected */
        $expected = $fixture->expected;

        Lre::assertDigits(Clustering::transitivity($graph), $expected['transitivity'], "{$name}: transitivity", digits: 12);
        Lre::assertDigits(Clustering::averageCoefficient($graph), $expected['average_clustering'], "{$name}: average clustering", digits: 12);
        self::assertSame($expected['triangle_total'], Clustering::triangleCount($graph), "{$name}: total triangles");
    }

    #[DataProvider('fixtures')]
    public function test_core_numbers_agree_with_networkx(string $name): void
    {
        $fixture = GraphFixture::load($name);
        $cores = KCore::coreNumbers($fixture->graph());

        foreach ($fixture->expectedVector('core_number') as $node => $expected) {
            self::assertSame((int) $expected, $cores[$node], "{$name}: core number of {$node}");
        }
    }

    /**
     * A star is bipartite, which is where plain power iteration oscillates
     * forever. It also has a closed form: the hub's score is 1/sqrt(2) and
     * each of the n-1 leaves gets 1/sqrt(2(n-1)).
     */
    public function test_eigenvector_centrality_converges_on_a_bipartite_graph(): void
    {
        $graph = Graph::undirected(10, array_map(static fn (int $l): array => [0, $l], range(1, 9)));
        $scores = (new Eigenvector())->of($graph);

        self::assertEqualsWithDelta(1.0 / sqrt(2.0), $scores[0], 1.0e-9, 'the hub');

        for ($leaf = 1; $leaf <= 9; $leaf++) {
            self::assertEqualsWithDelta(1.0 / sqrt(18.0), $scores[$leaf], 1.0e-9, "leaf {$leaf}");
        }
    }

    /**
     * A diverging Katz series produces inf, then inf/inf, and hands back a
     * vector of NaN with no clue as to why. On a weighted graph, where
     * lambda_max is large, that is the common case rather than an edge case.
     */
    public function test_katz_refuses_an_alpha_that_would_diverge(): void
    {
        $graph = GraphFixture::load('zachary')->graph();
        $critical = Katz::criticalAlpha($graph);

        self::assertGreaterThan(0.0, $critical);

        $this->expectException(InvalidArgument::class);

        (new Katz($critical * 1.5))->of($graph);
    }

    public function test_the_spectral_radius_of_shapes_with_a_known_answer(): void
    {
        // A complete graph on n nodes has lambda_max = n - 1.
        $edges = [];
        for ($i = 0; $i < 6; $i++) {
            for ($j = $i + 1; $j < 6; $j++) {
                $edges[] = [$i, $j];
            }
        }
        self::assertEqualsWithDelta(5.0, Katz::spectralRadius(Graph::undirected(6, $edges)), 1.0e-9);

        // A star on n nodes has lambda_max = sqrt(n - 1).
        $star = Graph::undirected(10, array_map(static fn (int $l): array => [0, $l], range(1, 9)));
        self::assertEqualsWithDelta(3.0, Katz::spectralRadius($star), 1.0e-9);

        self::assertSame(0.0, Katz::spectralRadius(Graph::undirected(5)));
    }

    /**
     * The two clustering summaries are different measures and are routinely
     * mistaken for each other. A star makes the gap unmissable: no triangles
     * anywhere, so both are zero -- while on a graph of two triangles sharing
     * a node they diverge.
     */
    public function test_average_clustering_and_transitivity_are_not_the_same_measure(): void
    {
        // Bowtie: two triangles sharing node 0.
        $graph = Graph::undirected(5, [[0, 1], [1, 2], [0, 2], [0, 3], [3, 4], [0, 4]]);

        // Nodes 1..4 each sit in one triangle with degree 2: coefficient 1.
        // Node 0 has degree 4, six pairs, two of them joined: 1/3.
        self::assertEqualsWithDelta((4 * 1.0 + 1 / 3) / 5, Clustering::averageCoefficient($graph), 1.0e-12);

        // 2 triangles, and 4 + 4*1 = 10 connected triples.
        self::assertEqualsWithDelta(3.0 * 2 / 10, Clustering::transitivity($graph), 1.0e-12);
    }

    public function test_k_core_prunes_the_periphery(): void
    {
        // A triangle with a tail: 0-1-2 joined, 3 hanging off 0, 4 off 3.
        $graph = Graph::undirected(5, [[0, 1], [1, 2], [0, 2], [0, 3], [3, 4]]);

        self::assertSame([2, 2, 2, 1, 1], KCore::coreNumbers($graph));
        self::assertSame(2, KCore::degeneracy($graph));
        self::assertSame([0, 1, 2], KCore::nodesInCore($graph, 2));
        self::assertSame([0, 1, 2, 3, 4], KCore::nodesInCore($graph, 1));
        self::assertSame([], KCore::nodesInCore($graph, 3));
    }

    public function test_the_measures_of_an_empty_graph_are_empty(): void
    {
        $empty = Graph::undirected(0);

        self::assertSame([], (new Eigenvector())->of($empty));
        self::assertSame([], Harmonic::of($empty));
        self::assertSame([], (new Katz())->of($empty));
        self::assertSame([], Clustering::triangles($empty));
        self::assertSame([], Clustering::coefficients($empty));
        self::assertSame([], KCore::coreNumbers($empty));
        self::assertSame(0.0, Clustering::transitivity($empty));
        self::assertSame(0.0, Clustering::averageCoefficient($empty));
    }

    public function test_a_graph_with_no_edges_has_no_structure_to_measure(): void
    {
        $graph = Graph::undirected(4);

        self::assertSame([0, 0, 0, 0], Clustering::triangles($graph));
        self::assertSame([0, 0, 0, 0], KCore::coreNumbers($graph));
        self::assertSame([0.0, 0.0, 0.0, 0.0], Harmonic::of($graph));
        self::assertSame(0.0, Clustering::transitivity($graph));
    }
}
