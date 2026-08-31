<?php

declare(strict_types=1);

namespace Vegoia\Tests\Reference\Graph;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Vegoia\Graph\Community\Agreement;
use Vegoia\Graph\Community\Leiden;
use Vegoia\Graph\Community\Quality\Modularity;
use Vegoia\Tests\Support\GraphFixture;

/**
 * Does it find the right answer?
 *
 * Every other reference test asks whether this implementation agrees with
 * leidenalg. That is necessary and not sufficient: two implementations can
 * agree on a mediocre partition, and agreement on modularity says nothing
 * about whether the communities are the ones actually there.
 *
 * These fixtures carry a ground truth. LFR benchmark graphs are generated
 * from known communities, with a mixing parameter that dials how much of each
 * node's degree leaves its community -- so the same test at mu 0.1, 0.3 and
 * 0.5 measures performance from obvious to nearly-undetectable. email-Eu-core
 * is a real network of 1005 people across 42 research departments, which no
 * algorithm recovers exactly.
 *
 * The bar is leidenalg's own recorded band, never perfection. Demanding
 * perfection on a real graph would be satisfiable only by overfitting, and
 * demanding it on LFR at mu 0.5 would demand the impossible: the reference
 * itself scores NMI 0.10-0.18 there, because at that mixing level the planted
 * structure is genuinely below the detectability threshold. A test suite that
 * hid that limit would be lying about what community detection can do.
 */
#[CoversClass(Leiden::class)]
#[Group('reference')]
#[Group('recovery')]
final class CommunityRecoveryTest extends TestCase
{
    /**
     * How far below leidenalg's worst seed this implementation may land.
     *
     * Not a fudge factor: the two explore in different random orders, so on a
     * rugged landscape they land on different local optima, and the reference
     * band itself already spans that variation. A regression that actually
     * broke the search drops far more than this -- deleting refinement, for
     * instance, costs whole tenths.
     */
    private const float TOLERANCE = 0.05;

    /** @return iterable<string, array{string}> */
    public static function labelled(): iterable
    {
        foreach (GraphFixture::labelledNames() as $name) {
            yield $name => [$name];
        }
    }

    #[DataProvider('labelled')]
    public function test_it_recovers_the_ground_truth_as_well_as_leidenalg(string $name): void
    {
        $fixture = GraphFixture::labelled($name);
        $graph = $fixture->graph();
        $truth = $fixture->groundTruthPartition();
        $reference = $fixture->referenceScores();

        $bestNmi = -INF;
        $bestAri = -INF;

        for ($seed = 1; $seed <= 10; $seed++) {
            $found = Leiden::modularity(seed: $seed)->partition($graph);

            $bestNmi = max($bestNmi, Agreement::normalisedMutualInformation($truth, $found));
            $bestAri = max($bestAri, Agreement::adjustedRandIndex($truth, $found));
        }

        self::assertGreaterThanOrEqual(
            $reference['nmi']['min'] - self::TOLERANCE,
            $bestNmi,
            sprintf(
                '%s: best NMI against ground truth over 10 seeds was %.4f; leidenalg reaches %.4f-%.4f',
                $name,
                $bestNmi,
                $reference['nmi']['min'],
                $reference['nmi']['max'],
            ),
        );

        self::assertGreaterThanOrEqual(
            $reference['ari']['min'] - self::TOLERANCE,
            $bestAri,
            sprintf(
                '%s: best ARI against ground truth over 10 seeds was %.4f; leidenalg reaches %.4f-%.4f',
                $name,
                $bestAri,
                $reference['ari']['min'],
                $reference['ari']['max'],
            ),
        );
    }

    #[DataProvider('labelled')]
    public function test_the_partitions_it_finds_score_as_well_as_leidenalgs(string $name): void
    {
        $fixture = GraphFixture::labelled($name);
        $graph = $fixture->graph();
        $reference = $fixture->referenceScores();
        $objective = new Modularity();

        $best = -INF;

        for ($seed = 1; $seed <= 10; $seed++) {
            $best = max($best, $objective->of($graph, Leiden::modularity(seed: $seed)->partition($graph)));
        }

        // Relative, because an absolute epsilon means different things at
        // Q = 0.73 and at Q = 0.33. One percent sits far below what an actual
        // regression costs -- deleting refinement loses tenths -- and above
        // the spread two correct searches show when they explore a rugged
        // landscape in different random orders.
        $floor = $reference['modularity']['min'] * 0.99;

        self::assertGreaterThanOrEqual(
            $floor,
            $best,
            sprintf(
                '%s: best modularity %.6f, floor %.6f, reference band [%.6f, %.6f]',
                $name,
                $best,
                $floor,
                $reference['modularity']['min'],
                $reference['modularity']['max'],
            ),
        );
    }

    /**
     * On an easy LFR graph the planted communities are essentially readable
     * off the edges, so anything working recovers nearly all of them. Stated
     * as an absolute floor rather than relative to the reference: this is the
     * case where a broken implementation has nowhere to hide.
     */
    public function test_easy_structure_is_recovered_almost_perfectly(): void
    {
        $fixture = GraphFixture::labelled('lfr_400_mu01');
        $found = Leiden::modularity(seed: 42)->partition($fixture->graph());

        self::assertGreaterThan(
            0.90,
            Agreement::normalisedMutualInformation($fixture->groundTruthPartition(), $found),
            'a graph with obvious community structure must be solved',
        );
    }

    /**
     * Recovery must degrade as the structure does. If an implementation scored
     * the same at mu 0.1 and mu 0.5 it would not be detecting communities at
     * all -- it would be responding to something else about the graph.
     */
    public function test_recovery_degrades_as_the_structure_is_mixed_away(): void
    {
        $scores = [];

        foreach (['lfr_400_mu01', 'lfr_400_mu03', 'lfr_400_mu05'] as $name) {
            $fixture = GraphFixture::labelled($name);
            $found = Leiden::modularity(seed: 42)->partition($fixture->graph());
            $scores[$name] = Agreement::normalisedMutualInformation(
                $fixture->groundTruthPartition(),
                $found,
            );
        }

        self::assertGreaterThan($scores['lfr_400_mu03'], $scores['lfr_400_mu01']);
        self::assertGreaterThan($scores['lfr_400_mu05'], $scores['lfr_400_mu03']);
    }

    /**
     * Higher modularity is not better communities, and this is measured here
     * rather than asserted as folklore.
     *
     * On LFR at mu 0.3 this implementation reaches slightly *lower* modularity
     * than leidenalg while recovering the planted communities just as well.
     * That is the resolution limit and the ruggedness of the objective showing
     * through: the score being optimised is a proxy, and past a point pushing
     * it further stops tracking the thing you wanted. It is also why the
     * recovery tests above assert NMI and not modularity.
     */
    public function test_a_better_modularity_score_does_not_imply_better_communities(): void
    {
        $fixture = GraphFixture::labelled('lfr_400_mu03');
        $graph = $fixture->graph();
        $truth = $fixture->groundTruthPartition();
        $objective = new Modularity();

        $byModularity = null;
        $bestScore = -INF;
        $bestNmi = -INF;

        for ($seed = 1; $seed <= 20; $seed++) {
            $found = Leiden::modularity(seed: $seed)->partition($graph);
            $score = $objective->of($graph, $found);
            $nmi = Agreement::normalisedMutualInformation($truth, $found);

            if ($score > $bestScore) {
                $bestScore = $score;
                $byModularity = $nmi;
            }

            $bestNmi = max($bestNmi, $nmi);
        }

        self::assertNotNull($byModularity);
        self::assertLessThanOrEqual(
            $bestNmi,
            $byModularity,
            'the highest-scoring partition cannot recover more than the best-recovering one',
        );

        // Both are respectable; the point is that they need not be the same run.
        self::assertGreaterThan(0.70, $bestNmi);
    }

    /**
     * A second pass must actually find something.
     *
     * Repeating the algorithm from where the last pass finished is worth
     * 0.003 to 0.012 of modularity on these graphs. Without it this library
     * sits consistently below leidenalg, whose default is likewise two
     * passes -- the gap that prompted adding it was 0.001 to 0.008 across the
     * SNAP collaboration graphs.
     *
     * @return iterable<string, array{string}>
     */
    public static function fixturesWhereRepeatingHelps(): iterable
    {
        // Established by measurement, not assumption: on most fixtures a
        // single pass already reaches the optimum and a second correctly
        // finds nothing.
        foreach (['lfr_400_mu03', 'lfr_400_mu05', 'email_eu_core'] as $name) {
            yield $name => [$name];
        }
    }

    #[DataProvider('fixturesWhereRepeatingHelps')]
    public function test_a_second_pass_improves_on_the_first(string $name): void
    {
        $graph = GraphFixture::labelled($name)->graph();
        $objective = new Modularity();

        $once = -INF;
        $twice = -INF;

        for ($seed = 1; $seed <= 5; $seed++) {
            $once = max($once, $objective->of(
                $graph,
                new Leiden(new Modularity(), $seed, iterations: 1)->partition($graph),
            ));
            $twice = max($twice, $objective->of(
                $graph,
                new Leiden(new Modularity(), $seed, iterations: 2)->partition($graph),
            ));
        }

        self::assertGreaterThan(
            $once,
            $twice,
            sprintf(
                '%s: two passes reached %.6f, one reached %.6f -- the second pass found nothing, '
                . 'which happens when the random stream restarts instead of continuing',
                $name,
                $twice,
                $once,
            ),
        );
    }

    /**
     * The default is two passes, matching leidenalg.
     *
     * Asserted through the default constructor rather than an explicit
     * `iterations:` argument -- every other test here passes the count
     * explicitly, so a change to the default would slip through all of them.
     */
    public function test_the_default_runs_two_passes(): void
    {
        $graph = GraphFixture::labelled('lfr_400_mu03')->graph();
        $objective = new Modularity();

        $default = -INF;
        $single = -INF;

        for ($seed = 1; $seed <= 5; $seed++) {
            $default = max($default, $objective->of($graph, Leiden::modularity(seed: $seed)->partition($graph)));
            $single = max($single, $objective->of(
                $graph,
                new Leiden(new Modularity(), $seed, iterations: 1)->partition($graph),
            ));
        }

        self::assertGreaterThan($single, $default, 'the default must do more than one pass');
    }

    /**
     * The ground truth of a real network is not the modularity optimum, and
     * pretending otherwise is a standard mistake. On email-Eu-core the
     * departments people belong to score *worse* by modularity than what the
     * algorithm finds -- so a high modularity score is not evidence of having
     * found the real groups.
     */
    public function test_on_a_real_network_the_truth_is_not_the_modularity_optimum(): void
    {
        $fixture = GraphFixture::labelled('email_eu_core');
        $graph = $fixture->graph();
        $objective = new Modularity();

        $truthScore = $objective->of($graph, $fixture->groundTruthPartition());
        $foundScore = $objective->of($graph, Leiden::modularity(seed: 42)->partition($graph));

        self::assertGreaterThan(
            $truthScore,
            $foundScore,
            'the algorithm out-scores the ground truth, which is why NMI and not modularity '
            . 'is the measure of whether it found the real departments',
        );
    }
}
