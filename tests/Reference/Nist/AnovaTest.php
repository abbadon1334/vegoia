<?php

declare(strict_types=1);

namespace Vegoia\Tests\Reference\Nist;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Vegoia\Stats\OneWayAnova;
use Vegoia\Tests\Support\AttainableAccuracy;
use Vegoia\Tests\Support\Lre;
use Vegoia\Tests\Support\NistAnova;

/**
 * One-way ANOVA against NIST certified values.
 *
 * The SmLs series is the point of this file. Each dataset holds the same
 * structure -- the certified F is exactly 21 for all of them -- at growing
 * digit counts, so the series measures precisely how fast an implementation
 * degrades as the signal shrinks relative to the values carrying it. The
 * textbook computational formula for the sums of squares is hopeless by
 * SmLs04; recentring the observations first, which ANOVA's invariance under
 * translation permits, matches SciPy on all of them.
 *
 * @see https://www.itl.nist.gov/div898/strd/anova/anova.html
 */
#[CoversClass(OneWayAnova::class)]
#[Group('reference')]
#[Group('nist')]
final class AnovaTest extends TestCase
{
    private const string CEILINGS = 'nist/attainable_anova.json';

    /** @return iterable<string, array{string}> */
    public static function datasets(): iterable
    {
        foreach (['SiRstv', 'AtmWtAg', 'SmLs01', 'SmLs04', 'SmLs07'] as $name) {
            yield $name => [$name];
        }
    }

    #[DataProvider('datasets')]
    public function test_f_statistic_matches_the_certified_value(string $name): void
    {
        $set = NistAnova::load($name);

        Lre::assertDigits(
            OneWayAnova::of($set->groups)->fStatistic,
            $set->fStatistic,
            "{$name}: F statistic",
            AttainableAccuracy::required($name, 'fStatistic', self::CEILINGS),
        );
    }

    #[DataProvider('datasets')]
    public function test_r_squared_matches_the_certified_value(string $name): void
    {
        $set = NistAnova::load($name);

        Lre::assertDigits(
            OneWayAnova::of($set->groups)->rSquared,
            $set->rSquared,
            "{$name}: R-squared",
            AttainableAccuracy::required($name, 'rSquared', self::CEILINGS),
        );
    }

    #[DataProvider('datasets')]
    public function test_residual_standard_deviation_matches(string $name): void
    {
        $set = NistAnova::load($name);

        Lre::assertDigits(
            OneWayAnova::of($set->groups)->residualStandardDeviation,
            $set->residualStandardDeviation,
            "{$name}: residual standard deviation",
            AttainableAccuracy::required($name, 'residualStandardDeviation', self::CEILINGS),
        );
    }

    #[DataProvider('datasets')]
    public function test_it_reports_the_shape_of_the_design(string $name): void
    {
        $set = NistAnova::load($name);
        $anova = OneWayAnova::of($set->groups);

        self::assertSame($set->betweenDegreesOfFreedom, $anova->betweenDegreesOfFreedom);
        self::assertSame($set->withinDegreesOfFreedom, $anova->withinDegreesOfFreedom);
        self::assertSame(count($set->groups), $anova->groups);
        self::assertSame(
            $set->betweenDegreesOfFreedom + $set->withinDegreesOfFreedom + 1,
            $anova->observations,
        );
    }

    /**
     * The certified F for every SmLs dataset is exactly 21. They differ only
     * in how many digits the observations carry, so the series isolates
     * numerical degradation from everything else.
     */
    public function test_the_smls_series_certifies_the_same_answer_at_every_difficulty(): void
    {
        foreach (['SmLs01', 'SmLs04', 'SmLs07'] as $name) {
            self::assertSame(21.0, NistAnova::load($name)->fStatistic, "{$name} is certified at F = 21");
        }

        // Benign: exactly right. Hardest: the limit of the arithmetic, which
        // the ceilings file records SciPy hitting too.
        self::assertSame(21.0, OneWayAnova::of(NistAnova::load('SmLs01')->groups)->fStatistic);
        self::assertEqualsWithDelta(21.0, OneWayAnova::of(NistAnova::load('SmLs07')->groups)->fStatistic, 1.0e-3);
    }

    public function test_grouping_by_label_matches_grouping_by_hand(): void
    {
        $set = NistAnova::load('SiRstv');

        $values = [];
        $labels = [];

        foreach ($set->groups as $index => $group) {
            foreach ($group as $value) {
                $values[] = $value;
                $labels[] = "g{$index}";
            }
        }

        self::assertSame(
            OneWayAnova::of($set->groups)->fStatistic,
            OneWayAnova::grouped($values, $labels)->fStatistic,
        );
    }
}
