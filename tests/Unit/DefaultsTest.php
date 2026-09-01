<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit;

use PHPUnit\Framework\TestCase;
use Vegoia\Exception\DidNotConverge;
use Vegoia\Graph\Centrality\Eigenvector;
use Vegoia\Graph\Centrality\Hits;
use Vegoia\Graph\Centrality\Katz;
use Vegoia\Graph\Centrality\PageRank;
use Vegoia\Graph\Community\Leiden;
use Vegoia\Graph\Community\Quality\ConstantPotts;
use Vegoia\Graph\Community\Quality\ErdosRenyiPotts;
use Vegoia\Graph\Community\Quality\Modularity;
use Vegoia\Stats\Descriptive;
use Vegoia\Stats\Regression\LeastSquares;
use Vegoia\Tests\Support\GraphFixture;

/**
 * The defaults are part of the interface, and nothing was checking them.
 *
 * Every test in this suite passes its arguments explicitly, which is good
 * practice and left a blind spot exactly the size of the signature: a
 * resolution of 1.0 could become 0.0, a seed of 0 could become 1, and the
 * whole suite would still pass while every caller who omitted the argument
 * got a different answer. Infection found 27 of these.
 *
 * Each one is pinned twice: the default must equal the value stated
 * explicitly, and it must differ from the neighbouring value a mutation would
 * put there. The first alone would be satisfied by any default at all.
 */
final class DefaultsTest extends TestCase
{
    public function test_the_modularity_resolution_defaults_to_one(): void
    {
        $graph = GraphFixture::load('zachary')->graph();
        $partition = Leiden::modularity(seed: 42)->partition($graph);

        self::assertSame(
            new Modularity(1.0)->of($graph, $partition),
            new Modularity()->of($graph, $partition),
        );

        self::assertNotSame(
            new Modularity(0.0)->of($graph, $partition),
            new Modularity()->of($graph, $partition),
            'a resolution of zero is a different objective, not the same one',
        );
    }

    public function test_the_erdos_renyi_resolution_defaults_to_one(): void
    {
        $graph = GraphFixture::load('lesmis')->graph();
        $partition = Leiden::modularity(seed: 7)->partition($graph);

        self::assertSame(
            new ErdosRenyiPotts(1.0)->of($graph, $partition),
            new ErdosRenyiPotts()->of($graph, $partition),
        );

        self::assertNotSame(
            new ErdosRenyiPotts(0.0)->of($graph, $partition),
            new ErdosRenyiPotts()->of($graph, $partition),
        );
    }

    /**
     * The four factories, each of which names a resolution and a seed.
     *
     * Both matter and for different reasons: the resolution decides what
     * counts as a community, and the seed decides which of several equally
     * good answers you get. A library whose default seed moved between
     * releases would be irreproducible without anybody's tests failing.
     */
    public function test_the_leiden_factories_default_to_resolution_one_and_seed_zero(): void
    {
        $graph = GraphFixture::load('lesmis')->graph();

        $factories = [
            'modularity' => [
                static fn (): Leiden => Leiden::modularity(),
                static fn (): Leiden => Leiden::modularity(1.0, 0),
                static fn (): Leiden => Leiden::modularity(0.05, 0),
            ],
            'constantPotts' => [
                static fn (): Leiden => Leiden::constantPotts(),
                static fn (): Leiden => Leiden::constantPotts(1.0, 0),
                static fn (): Leiden => Leiden::constantPotts(0.05, 0),
            ],
            'erdosRenyiPotts' => [
                static fn (): Leiden => Leiden::erdosRenyiPotts(),
                static fn (): Leiden => Leiden::erdosRenyiPotts(1.0, 0),
                static fn (): Leiden => Leiden::erdosRenyiPotts(0.05, 0),
            ],
            'reichardtBornholdt' => [
                static fn (): Leiden => Leiden::reichardtBornholdt(),
                static fn (): Leiden => Leiden::reichardtBornholdt(1.0, 0),
                static fn (): Leiden => Leiden::reichardtBornholdt(0.05, 0),
            ],
        ];

        foreach ($factories as $name => [$default, $explicit, $other]) {
            self::assertSame(
                $explicit()->partition($graph)->membership(),
                $default()->partition($graph)->membership(),
                "{$name}: the default is not resolution 1.0 with seed 0",
            );

            self::assertNotSame(
                $other()->partition($graph)->membership(),
                $default()->partition($graph)->membership(),
                "{$name}: a different resolution must give a different partition, or this "
                . 'test cannot tell one default from another',
            );
        }
    }

    /**
     * The seed default, separately, because the resolution check above would
     * pass with any seed as long as it were used consistently.
     */
    public function test_the_leiden_seed_defaults_to_zero(): void
    {
        // Petersen, not lesmis. On lesmis every seed finds the same partition,
        // so the test would have passed whatever the default was -- which the
        // second assertion below caught when this was written against it.
        // Petersen is small, symmetric and has several equally good answers,
        // which is precisely what makes the seed visible.
        $graph = GraphFixture::load('petersen')->graph();

        $default = Leiden::modularity(1.0)->partition($graph)->membership();

        self::assertSame(
            Leiden::modularity(1.0, 0)->partition($graph)->membership(),
            $default,
            'the default seed is not 0',
        );

        foreach ([1, -1] as $seed) {
            self::assertNotSame(
                Leiden::modularity(1.0, $seed)->partition($graph)->membership(),
                $default,
                "seed {$seed} reproduces seed 0 here, so this test cannot see the default move",
            );
        }
    }

    /** The constructor's own seed default, which the factories go through. */
    public function test_the_leiden_constructor_seed_defaults_to_zero(): void
    {
        $graph = GraphFixture::load('petersen')->graph();
        $default = new Leiden(new Modularity())->partition($graph)->membership();

        self::assertSame(new Leiden(new Modularity(), 0)->partition($graph)->membership(), $default);
        self::assertNotSame(new Leiden(new Modularity(), 1)->partition($graph)->membership(), $default);
    }

    /** The CPM resolution, which decides the density a community must reach. */
    public function test_the_constant_potts_resolution_defaults_to_one(): void
    {
        $graph = GraphFixture::load('lesmis')->graph();
        $partition = Leiden::modularity(seed: 7)->partition($graph);

        self::assertSame(
            new ConstantPotts(1.0)->of($graph, $partition),
            new ConstantPotts()->of($graph, $partition),
        );

        self::assertNotSame(
            new ConstantPotts(0.05)->of($graph, $partition),
            new ConstantPotts()->of($graph, $partition),
        );
    }

    public function test_the_autocorrelation_lag_defaults_to_one(): void
    {
        $values = [];

        for ($i = 0; $i < 60; $i++) {
            $values[] = sin($i / 3.0) + $i / 40.0;
        }

        $sample = Descriptive::of($values);

        self::assertSame($sample->autocorrelation(1), $sample->autocorrelation());
        self::assertNotSame($sample->autocorrelation(2), $sample->autocorrelation());
    }

    /**
     * A regression fits an intercept unless told otherwise, which is the
     * convention every statistical package follows and the one the NIST
     * datasets are certified against.
     */
    public function test_a_regression_includes_an_intercept_by_default(): void
    {
        $predictors = [[1.0], [2.0], [3.0], [4.0], [5.0]];
        $response = [2.1, 3.9, 6.2, 7.8, 10.1];

        $default = LeastSquares::fit($predictors, $response);

        self::assertTrue($default->hasIntercept);
        self::assertSame(LeastSquares::fit($predictors, $response, true)->coefficients, $default->coefficients);
        self::assertNotSame(
            LeastSquares::fit($predictors, $response, false)->coefficients,
            $default->coefficients,
        );
    }

    /**
     * The iteration ceilings are not defaults anybody passes, and the default
     * must not be what ends the loop -- an algorithm that stopped because it
     * ran out of iterations rather than because it settled is reporting
     * whatever it happened to be holding.
     *
     * Two halves. A ceiling of two is refused outright, which is how this is
     * checked now: starving the search used to return a plausible vector and
     * the test compared it with the converged one, which was a weaker
     * question and a worse contract. And the default reaches the same answer
     * a much larger ceiling does, which is what says the default is generous
     * rather than merely large.
     */
    public function test_the_iteration_ceiling_is_not_what_stops_the_search(): void
    {
        $graph = GraphFixture::load('lesmis')->graph();

        $converged = new PageRank()->of($graph);
        $generous = new PageRank(maxIterations: 100_000)->of($graph);

        foreach ($converged as $node => $rank) {
            self::assertEqualsWithDelta($generous[$node], $rank, 1.0e-15, "node {$node}");
        }

        $this->expectException(DidNotConverge::class);
        $this->expectExceptionMessageMatches('/PageRank did not converge in 2 iterations/');

        new PageRank(maxIterations: 2)->of($graph);
    }

    /**
     * Every iterative method refuses rather than returning what it was
     * holding when the iterations ran out.
     *
     * Katz already did this for its own divergence case, and the rest
     * disagreed with it: same class of failure, opposite answer. They agree
     * now, and this is the test that keeps them agreeing.
     */
    public function test_no_iterative_method_returns_an_unsettled_answer(): void
    {
        $graph = GraphFixture::load('lesmis')->graph();
        $directed = GraphFixture::directed('hub_and_authority')->directedGraph();

        $starved = [
            'PageRank' => static fn () => new PageRank(maxIterations: 2)->of($graph),
            'Eigenvector' => static fn () => new Eigenvector(maxIterations: 2)->of($graph),
            'HITS' => static fn () => new Hits(maxIterations: 2)->of($directed),
            // Below 1 / lambda_max for Les Misérables, which is 0.0154 --
            // above it Katz refuses for a different and equally good reason,
            // and this test is about the iteration ceiling.
            'Katz' => static fn () => new Katz(alpha: 0.014, maxIterations: 2)->of($graph),
        ];

        foreach ($starved as $name => $call) {
            try {
                $call();
                self::fail("{$name} returned an answer after two iterations instead of refusing");
            } catch (DidNotConverge $e) {
                self::assertStringContainsString(
                    'did not converge in 2 iterations',
                    $e->getMessage(),
                    $name,
                );
            }
        }
    }
}
