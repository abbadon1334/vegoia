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
use Vegoia\Graph\Community\Quality\QualityFunction;
use Vegoia\Graph\Graph;
use Vegoia\Graph\Partition;
use Vegoia\Tests\Support\GraphFixture;

/**
 * The optimised path and the general path must agree, exactly.
 *
 * Leiden inlines the gain and connectivity formulas for its two built-in
 * objectives, because going through the interface once per candidate
 * community per node per pass dominates the profile. That leaves two copies
 * of each formula: the one inlined in the algorithm, and the one in the
 * QualityFunction it bypasses.
 *
 * Two copies of a formula drift. Worse, they drift silently -- the inlined
 * path is what every benchmark and almost every test exercises, so an error
 * introduced only there is invisible, and an error introduced only in the
 * interface method is invisible to everything except a custom objective,
 * which is to say a user's code.
 *
 * Wrapping an objective in a delegating class defeats the `::class` check
 * that selects the fast path, so the same run can be performed both ways.
 * Identical seeds must then give identical partitions, because the two paths
 * are supposed to be the same arithmetic.
 */
#[CoversClass(Leiden::class)]
#[Group('reference')]
final class LeidenPathEquivalenceTest extends TestCase
{
    /** @return iterable<string, array{string}> */
    public static function fixtures(): iterable
    {
        foreach (GraphFixture::names() as $name) {
            yield $name => [$name];
        }
    }

    #[DataProvider('fixtures')]
    public function test_modularity_gives_the_same_partition_through_either_path(string $name): void
    {
        $graph = GraphFixture::load($name)->graph();

        foreach ([0.5, 1.0, 2.0] as $resolution) {
            for ($seed = 1; $seed <= 5; $seed++) {
                $fast = new Leiden(new Modularity($resolution), $seed)->partition($graph);
                $general = new Leiden(self::opaque(new Modularity($resolution)), $seed)->partition($graph);

                self::assertTrue(
                    $fast->equals($general),
                    "{$name}: inlined and interface paths disagree at resolution {$resolution}, seed {$seed} "
                    . "({$fast->count()} vs {$general->count()} communities)",
                );
            }
        }
    }

    #[DataProvider('fixtures')]
    public function test_constant_potts_gives_the_same_partition_through_either_path(string $name): void
    {
        $graph = GraphFixture::load($name)->graph();

        foreach ([0.05, 0.5] as $resolution) {
            for ($seed = 1; $seed <= 5; $seed++) {
                $fast = new Leiden(new ConstantPotts($resolution), $seed)->partition($graph);
                $general = new Leiden(self::opaque(new ConstantPotts($resolution)), $seed)->partition($graph);

                self::assertTrue(
                    $fast->equals($general),
                    "{$name}: inlined and interface paths disagree at resolution {$resolution}, seed {$seed}",
                );
            }
        }
    }

    public function test_a_custom_objective_is_actually_taking_the_general_path(): void
    {
        // Guards the guard: if the wrapper were somehow recognised as a
        // built-in, every assertion above would compare the fast path with
        // itself and prove nothing.
        $wrapper = self::opaque(new Modularity());

        self::assertNotInstanceOf(Modularity::class, $wrapper);
        self::assertNotSame(Modularity::class, $wrapper::class);
        self::assertNotSame(ConstantPotts::class, $wrapper::class);
    }

    /** An objective that behaves identically but is not one of the known classes. */
    private static function opaque(QualityFunction $inner): QualityFunction
    {
        return new class ($inner) implements QualityFunction {
            public function __construct(private readonly QualityFunction $inner)
            {
            }

            public function resolution(): float
            {
                return $this->inner->resolution();
            }

            public function boundTo(Graph $graph): QualityFunction
            {
                // Stay opaque after binding, or the wrapper would hand Leiden
                // back the very class the fast path recognises.
                return new self($this->inner->boundTo($graph));
            }

            public function of(Graph $graph, Partition $partition): float
            {
                return $this->inner->of($graph, $partition);
            }

            public function gain(
                float $weightToCommunity,
                float $nodeStrength,
                float $nodeSize,
                float $communityStrength,
                float $communitySize,
                float $totalEndpointWeight,
            ): float {
                return $this->inner->gain(
                    $weightToCommunity,
                    $nodeStrength,
                    $nodeSize,
                    $communityStrength,
                    $communitySize,
                    $totalEndpointWeight,
                );
            }

            public function measure(float $strength, float $size): float
            {
                return $this->inner->measure($strength, $size);
            }

            public function connectivityThreshold(
                float $partMeasure,
                float $subsetMeasure,
                float $totalEndpointWeight,
            ): float {
                return $this->inner->connectivityThreshold(
                    $partMeasure,
                    $subsetMeasure,
                    $totalEndpointWeight,
                );
            }
        };
    }
}
