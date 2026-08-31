<?php

declare(strict_types=1);

namespace Vegoia\Tests\Reference\Graph;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Vegoia\Graph\Community\Leiden;
use Vegoia\Graph\Community\Quality\ConstantPotts;
use Vegoia\Graph\Community\Quality\Modularity;
use Vegoia\Tests\Support\GraphFixture;

/**
 * The Constant Potts model, against leidenalg's own runs at three resolutions.
 *
 * These bands have been in the fixtures since they were first generated and
 * nothing read them, which mutation testing eventually made visible. They are
 * worth reading: CPM is the objective to reach for when modularity's
 * resolution limit is the problem -- it has no null model, so its resolution
 * means the same thing on a graph of fifty nodes and one of fifty thousand,
 * and a community is simply a group internally denser than the resolution.
 *
 * What they record is what leidenalg's CPM produced over fifty seeds at each
 * resolution: how many communities, and the modularity of the result. The
 * second is incidental rather than the objective -- CPM does not optimise
 * modularity and a better CPM partition can score worse by it -- so this file
 * asks less of the modularity than LeidenTest asks of the modularity
 * envelope, where the maximum is the right bar because it is the thing being
 * maximised.
 */
#[CoversClass(Leiden::class)]
#[CoversClass(ConstantPotts::class)]
#[Group('reference')]
final class ConstantPottsTest extends TestCase
{
    private const int SEEDS = 10;

    /** @return iterable<string, array{string}> */
    public static function fixtures(): iterable
    {
        foreach (GraphFixture::names() as $name) {
            yield $name => [$name];
        }
    }

    /**
     * The number of communities must overlap what the reference found.
     *
     * Overlap, not containment: both sides are samples from a stochastic
     * search over a different number of seeds, and leidenalg's own band moved
     * between its 0.10 and 0.12 releases. Two searches that disagree about
     * whether a graph has 12 or 21 communities at a given resolution are
     * exploring the same landscape; two whose ranges do not meet at all are
     * not solving the same problem.
     */
    #[DataProvider('fixtures')]
    public function test_it_finds_as_many_communities_as_the_reference(string $name): void
    {
        $fixture = GraphFixture::load($name);
        $graph = $fixture->graph();

        foreach ($fixture->leidenConstantPottsBands() as $resolution => $band) {
            $counts = [];

            for ($seed = 1; $seed <= self::SEEDS; $seed++) {
                $counts[] = Leiden::constantPotts((float) $resolution, $seed)->partition($graph)->count();
            }

            $low = min($counts);
            $high = max($counts);

            self::assertLessThanOrEqual(
                $band['communities']['max'],
                $low,
                sprintf(
                    '%s at resolution %s: found %d-%d communities, the reference found %d-%d, and the '
                    . 'two ranges do not meet',
                    $name,
                    $resolution,
                    $low,
                    $high,
                    $band['communities']['min'],
                    $band['communities']['max'],
                ),
            );

            self::assertGreaterThanOrEqual(
                $band['communities']['min'],
                $high,
                sprintf(
                    '%s at resolution %s: found %d-%d communities, the reference found %d-%d, and the '
                    . 'two ranges do not meet',
                    $name,
                    $resolution,
                    $low,
                    $high,
                    $band['communities']['min'],
                    $band['communities']['max'],
                ),
            );
        }
    }

    /**
     * The modularity of the CPM partitions must reach the bottom of the
     * reference's own spread.
     *
     * The bottom and not the top, deliberately. Modularity is not what CPM is
     * optimising, so demanding the maximum would be demanding that this
     * implementation do better at a job it was not given -- and on lesmis at
     * resolution 0.5 the reference's best modularity comes from a seed whose
     * CPM quality is not its best either. Reaching the worst of fifty seeds
     * still fails loudly if CPM is broken: a partition that ignores the
     * resolution scores far below the band, not just under its top.
     */
    #[DataProvider('fixtures')]
    public function test_its_partitions_score_within_the_reference_spread(string $name): void
    {
        $fixture = GraphFixture::load($name);
        $graph = $fixture->graph();
        $modularity = new Modularity();

        foreach ($fixture->leidenConstantPottsBands() as $resolution => $band) {
            $best = -INF;

            for ($seed = 1; $seed <= self::SEEDS; $seed++) {
                $best = max($best, $modularity->of(
                    $graph,
                    Leiden::constantPotts((float) $resolution, $seed)->partition($graph),
                ));
            }

            self::assertGreaterThanOrEqual(
                $band['modularity']['min'] - 1.0e-9,
                $best,
                sprintf(
                    '%s at resolution %s: best modularity over %d seeds was %.6f, below the worst of the '
                    . "reference's fifty (%.6f); its band is [%.6f, %.6f]",
                    $name,
                    $resolution,
                    self::SEEDS,
                    $best,
                    $band['modularity']['min'],
                    $band['modularity']['min'],
                    $band['modularity']['max'],
                ),
            );
        }
    }

    /**
     * Optimising CPM must beat a partition that was optimising something else.
     *
     * The comparison the bands cannot make. Each fixture also carries the
     * partition leidenalg found at seed 42 while maximising modularity, scored
     * under CPM at three resolutions. A working CPM search, given the same
     * graph and the same resolution, has to do at least as well by its own
     * objective -- and this is checked with this library's own ConstantPotts
     * on both sides, so it is a statement about the search rather than about
     * agreeing with anybody's arithmetic.
     */
    #[DataProvider('fixtures')]
    public function test_optimising_the_objective_beats_a_partition_that_did_not(string $name): void
    {
        $fixture = GraphFixture::load($name);
        $graph = $fixture->graph();

        foreach ([0.05, 0.1, 0.5, 1.0] as $resolution) {
            $objective = new ConstantPotts($resolution);

            $found = -INF;

            for ($seed = 1; $seed <= self::SEEDS; $seed++) {
                $found = max($found, $objective->of(
                    $graph,
                    Leiden::constantPotts($resolution, $seed)->partition($graph),
                ));
            }

            $byModularity = $objective->of($graph, Leiden::modularity(seed: 42)->partition($graph));

            self::assertGreaterThanOrEqual(
                $byModularity - 1.0e-9,
                $found,
                sprintf(
                    '%s at resolution %.2f: optimising CPM scored %.6f, while a partition found by '
                    . 'optimising modularity scored %.6f under the same objective',
                    $name,
                    $resolution,
                    $found,
                    $byModularity,
                ),
            );
        }
    }

    /**
     * Raising the resolution cannot lower the number of communities.
     *
     * The property that gives CPM its name and its purpose: the resolution is
     * the density a group has to beat to be worth keeping separate, so asking
     * for more density can only split, never merge. An implementation that
     * ignored the parameter would pass every band test above by accident and
     * fail this one.
     */
    #[DataProvider('fixtures')]
    public function test_a_higher_resolution_never_finds_fewer_communities(string $name): void
    {
        $graph = GraphFixture::load($name)->graph();
        $previous = 0;

        foreach ([0.01, 0.05, 0.1, 0.5, 1.0] as $resolution) {
            $best = 0;

            for ($seed = 1; $seed <= self::SEEDS; $seed++) {
                $best = max($best, Leiden::constantPotts($resolution, $seed)->partition($graph)->count());
            }

            self::assertGreaterThanOrEqual(
                $previous,
                $best,
                sprintf('%s: resolution %.2f found fewer communities than the one below it', $name, $resolution),
            );

            $previous = $best;
        }
    }
}
