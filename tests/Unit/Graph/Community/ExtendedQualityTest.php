<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit\Graph\Community;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;
use Vegoia\Graph\Community\Quality\ErdosRenyiPotts;
use Vegoia\Graph\Community\Quality\KullbackLeibler;
use Vegoia\Graph\Community\Quality\Modularity;
use Vegoia\Graph\Community\Quality\Significance;
use Vegoia\Graph\Community\Quality\Surprise;
use Vegoia\Graph\Graph;
use Vegoia\Graph\Partition;
use Vegoia\Tests\Support\GraphFixture;
use Vegoia\Tests\Support\Lre;

/**
 * The objectives leidenalg offers beyond modularity and CPM.
 *
 * Two conventions differ from leidenalg's and are asserted rather than
 * quietly absorbed:
 *
 *   * leidenalg reports UNNORMALISED quality. Its modularity is ours times
 *     2m, and its RBER is ours times two. Ours matches the published formulas
 *     and is consistent with CPM, so the factor is applied here, in the open,
 *     instead of being folded into the implementation where nobody would find
 *     it when comparing against a Python result.
 *
 *   * Surprise and Significance count edges and node pairs, so weights have no
 *     meaning in them. A weighted graph is scored on its topology, which is
 *     what leidenalg does too.
 */
#[CoversClass(ErdosRenyiPotts::class)]
#[CoversClass(Surprise::class)]
#[CoversClass(Significance::class)]
#[CoversClass(KullbackLeibler::class)]
final class ExtendedQualityTest extends TestCase
{
    /** leidenalg reports RBER without the 1/2 the published formula carries. */
    private const float RBER_SCALE = 2.0;

    /** @return iterable<string, array{string, string, string}> */
    public static function rberProbes(): iterable
    {
        foreach (GraphFixture::names() as $name) {
            foreach (['singletons', 'single_community', 'leiden_seed42'] as $probe) {
                foreach (['0.5', '1.0', '2.0'] as $resolution) {
                    yield "{$name}/{$probe}/y={$resolution}" => [$name, $probe, $resolution];
                }
            }
        }
    }

    #[DataProvider('rberProbes')]
    public function test_erdos_renyi_potts_agrees_with_leidenalg(string $name, string $probe, string $resolution): void
    {
        $fixture = GraphFixture::load($name);
        $entry = self::probe($fixture, $probe);

        self::assertAccurate(
            self::RBER_SCALE * (new ErdosRenyiPotts((float) $resolution))->of(
                $fixture->graph(),
                Partition::fromMembership($entry['membership']),
            ),
            $entry['rber'][$resolution],
            "{$name}/{$probe}: RBER at resolution {$resolution}",
        );
    }

    /** @return iterable<string, array{string, string}> */
    public static function scoreProbes(): iterable
    {
        foreach (GraphFixture::names() as $name) {
            foreach (['singletons', 'single_community', 'leiden_seed42'] as $probe) {
                yield "{$name}/{$probe}" => [$name, $probe];
            }
        }
    }

    #[DataProvider('scoreProbes')]
    public function test_surprise_agrees_with_leidenalg(string $name, string $probe): void
    {
        $fixture = GraphFixture::load($name);
        $entry = self::probe($fixture, $probe);

        self::assertAccurate(
            (new Surprise())->of($fixture->graph(), Partition::fromMembership($entry['membership'])),
            $entry['surprise'],
            "{$name}/{$probe}: Surprise",
        );
    }

    #[DataProvider('scoreProbes')]
    public function test_significance_agrees_with_leidenalg(string $name, string $probe): void
    {
        $fixture = GraphFixture::load($name);
        $entry = self::probe($fixture, $probe);

        self::assertAccurate(
            (new Significance())->of($fixture->graph(), Partition::fromMembership($entry['membership'])),
            $entry['significance'],
            "{$name}/{$probe}: Significance",
        );
    }

    /**
     * RBER is CPM with the resolution scaled by the graph's density, so on a
     * graph of density p the two agree when RBER's resolution is 1/p times
     * CPM's. Verifying the relation rather than restating the formula catches
     * a density computed differently in the two places.
     */
    public function test_it_is_constant_potts_scaled_by_density(): void
    {
        $fixture = GraphFixture::load('zachary');
        $graph = $fixture->graph();
        $partition = Partition::fromMembership(
            self::probe($fixture, 'leiden_seed42')['membership']
        );

        $density = ErdosRenyiPotts::densityOf($graph);

        self::assertEqualsWithDelta(
            (new \Vegoia\Graph\Community\Quality\ConstantPotts($density))->of($graph, $partition),
            (new ErdosRenyiPotts(1.0))->of($graph, $partition),
            1.0e-12,
        );
    }

    public function test_density_of_degenerate_graphs(): void
    {
        self::assertSame(0.0, ErdosRenyiPotts::densityOf(Graph::undirected(0)));
        self::assertSame(0.0, ErdosRenyiPotts::densityOf(Graph::undirected(1)));
        self::assertSame(1.0, ErdosRenyiPotts::densityOf(Graph::undirected(2, [[0, 1]])));
    }

    /**
     * Both boundary cases arise for real: a clique has internal density 1 and
     * an isolated pair has 0, and the naive x*log(x) yields NaN at each.
     */
    public function test_kullback_leibler_handles_the_boundaries(): void
    {
        self::assertSame(0.0, KullbackLeibler::binary(0.5, 0.5), 'no divergence from itself');
        self::assertGreaterThan(0.0, KullbackLeibler::binary(0.9, 0.1));
        self::assertGreaterThan(0.0, KullbackLeibler::binary(0.1, 0.9));

        self::assertTrue(is_finite(KullbackLeibler::binary(1.0, 0.5)), 'a full community');
        self::assertTrue(is_finite(KullbackLeibler::binary(0.0, 0.5)), 'an empty one');

        // A reference distribution at a boundary has nothing to diverge from.
        self::assertSame(0.0, KullbackLeibler::binary(0.5, 0.0));
        self::assertSame(0.0, KullbackLeibler::binary(0.5, 1.0));
    }

    public function test_the_scores_of_a_graph_with_no_edges_are_zero(): void
    {
        $graph = Graph::undirected(5);
        $partition = Partition::singletons(5);

        self::assertSame(0.0, (new Surprise())->of($graph, $partition));
        self::assertSame(0.0, (new Significance())->of($graph, $partition));
        self::assertSame(0.0, (new ErdosRenyiPotts())->of($graph, $partition));
        self::assertSame(0.0, (new Modularity())->of($graph, $partition));
    }

    /**
     * Relative accuracy, except where the true value is zero.
     *
     * Several of these probes are exactly zero by construction -- RBER of a
     * single community at resolution 1 is m - p*C(n,2), which cancels exactly.
     * We return 0.0; leidenalg returns 2.3e-13 of accumulated rounding. A
     * relative measure calls that a total mismatch, when in fact our answer is
     * the better one, so near zero the comparison has to be absolute.
     */
    private static function assertAccurate(float $computed, float $certified, string $what): void
    {
        if (abs($certified) < 1.0e-9) {
            self::assertEqualsWithDelta(
                0.0,
                $computed,
                1.0e-9,
                "{$what}: expected zero (the reference carries {$certified} of rounding noise)",
            );

            return;
        }

        Lre::assertDigits($computed, $certified, $what, digits: 11);
    }

    /**
     * @return array{membership: list<int>, rber: array<string, float>, surprise: float, significance: float}
     */
    private static function probe(GraphFixture $fixture, string $probe): array
    {
        /** @var array{quality_probes: array<string, array{membership: list<int>, rber: array<string, float>, surprise: float, significance: float}>} $expected */
        $expected = $fixture->expected;

        return $expected['quality_probes'][$probe];
    }
}
