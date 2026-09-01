<?php

declare(strict_types=1);

namespace Vegoia\Tests\Reference\Stats;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Vegoia\Stats\Alternative;
use Vegoia\Stats\ChiSquaredTest;
use Vegoia\Stats\Continuity;
use Vegoia\Stats\KruskalWallis;
use Vegoia\Stats\MannWhitneyU;
use Vegoia\Stats\TTest;
use Vegoia\Tests\Support\Lre;
use Vegoia\Tests\Support\Paths;

/**
 * The four hypothesis tests, against SciPy.
 *
 * Two conventions here are worth reading before the assertions, because both
 * were verified against SciPy rather than taken from documentation, and both
 * are places where an implementation can look right and disagree.
 *
 * Yates' continuity correction is applied by default and only touches tables
 * with one degree of freedom. The shift is clamped to the difference it is
 * correcting: on a table where every |observed - expected| is 0.244, the
 * textbook `(|o - e| - 0.5)^2 / e` overshoots and reports 0.023 where the
 * answer is exactly zero.
 *
 * Mann-Whitney is generated with `method='asymptotic'` passed explicitly.
 * SciPy's `auto` switches between the exact and the asymptotic route on sample
 * size and on the presence of ties, and the two disagree materially, so a
 * fixture built with it would pin SciPy's heuristic rather than the
 * mathematics.
 *
 * @see tools/generate_hypothesis_fixtures.py
 */
#[CoversClass(ChiSquaredTest::class)]
#[CoversClass(TTest::class)]
#[CoversClass(MannWhitneyU::class)]
#[CoversClass(KruskalWallis::class)]
#[Group('reference')]
final class HypothesisTest extends TestCase
{
    /** @var array<string, mixed>|null */
    private static ?array $fixture = null;

    /** @return array<string, mixed> */
    private static function section(string $name): array
    {
        if (self::$fixture === null) {
            /** @var array<string, mixed> $decoded */
            $decoded = json_decode(
                (string) file_get_contents(Paths::fixture('stats/hypothesis.json')),
                true,
                flags: JSON_THROW_ON_ERROR,
            );
            self::$fixture = $decoded;
        }

        /** @var array<string, mixed> $out */
        $out = self::$fixture[$name];

        return $out;
    }

    /** @return iterable<string, array{string, array<string, mixed>}> */
    public static function tables(): iterable
    {
        /** @var array<string, array<string, mixed>> $entries */
        $entries = self::section('chi_squared');

        foreach ($entries as $name => $entry) {
            yield $name => [$name, $entry];
        }
    }

    /** @return iterable<string, array{string, array<string, mixed>}> */
    public static function samples(): iterable
    {
        /** @var array<string, array<string, mixed>> $entries */
        $entries = self::section('t');

        foreach ($entries as $name => $entry) {
            yield $name => [$name, $entry];
        }
    }

    /** @return iterable<string, array{string, array<string, mixed>}> */
    public static function rankSamples(): iterable
    {
        /** @var array<string, array<string, mixed>> $entries */
        $entries = self::section('mann_whitney');

        foreach ($entries as $name => $entry) {
            yield $name => [$name, $entry];
        }
    }

    /** @return iterable<string, array{string, array<string, mixed>}> */
    public static function groupSets(): iterable
    {
        /** @var array<string, array<string, mixed>> $entries */
        $entries = self::section('kruskal_wallis');

        foreach ($entries as $name => $entry) {
            yield $name => [$name, $entry];
        }
    }

    /** @param array<string, mixed> $entry */
    #[DataProvider('tables')]
    public function test_the_chi_squared_statistic_matches_scipy(string $name, array $entry): void
    {
        /**
         * @var array{
         *     table: list<list<int>>,
         *     corrected: array{statistic: float, p_value: float, degrees_of_freedom: int},
         *     uncorrected: array{statistic: float, p_value: float, degrees_of_freedom: int},
         *     expected: list<list<float>>
         * } $entry
         */
        $table = $entry['table'];

        foreach (['corrected' => Continuity::Corrected, 'uncorrected' => Continuity::Uncorrected] as $key => $mode) {
            $test = ChiSquaredTest::independence($table, $mode);
            /** @var array{statistic: float, p_value: float, degrees_of_freedom: int} $expected */
            $expected = $entry[$key];

            self::assertSame($expected['degrees_of_freedom'], $test->degreesOfFreedom, "{$name}: df");

            if ($expected['statistic'] === 0.0) {
                self::assertSame(0.0, $test->statistic, "{$name}: {$key} statistic");
            } else {
                Lre::assertDigits($test->statistic, $expected['statistic'], "{$name}: {$key} statistic", 13.0);
            }

            Lre::assertDigits($test->pValue(), $expected['p_value'], "{$name}: {$key} p-value", 12.0);
        }

        // The expected counts are the table's own margins, so they are the
        // part a caller can check the arithmetic against.
        $test = ChiSquaredTest::independence($table);

        foreach ($entry['expected'] as $row => $columns) {
            foreach ($columns as $column => $value) {
                Lre::assertDigits(
                    $test->expected[$row][$column],
                    $value,
                    "{$name}: expected count at ({$row}, {$column})",
                    14.0,
                );
            }
        }
    }

    /**
     * Yates only touches a table with one degree of freedom.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('tables')]
    public function test_the_correction_only_touches_two_by_two_tables(string $name, array $entry): void
    {
        /** @var array{table: list<list<int>>} $entry */
        $corrected = ChiSquaredTest::independence($entry['table'], Continuity::Corrected);
        $uncorrected = ChiSquaredTest::independence($entry['table'], Continuity::Uncorrected);

        if ($corrected->degreesOfFreedom > 1) {
            self::assertSame(
                $uncorrected->statistic,
                $corrected->statistic,
                "{$name}: the correction reached a table with more than one degree of freedom",
            );
        } else {
            self::assertLessThanOrEqual(
                $uncorrected->statistic,
                $corrected->statistic,
                "{$name}: the correction must move the statistic towards the null, never away",
            );
        }
    }

    /** @param array<string, mixed> $entry */
    #[DataProvider('samples')]
    public function test_the_t_statistics_match_scipy(string $name, array $entry): void
    {
        /**
         * @var array{
         *     x: list<float>, y: list<float>,
         *     student: array{statistic: float, p_value: float, degrees_of_freedom: float, confidence_interval: array{float, float}},
         *     welch: array{statistic: float, p_value: float, degrees_of_freedom: float, confidence_interval: array{float, float}}
         * } $entry
         */
        $variants = [
            'student' => TTest::student($entry['x'], $entry['y']),
            'welch' => TTest::welch($entry['x'], $entry['y']),
        ];

        foreach ($variants as $key => $test) {
            /** @var array{statistic: float, p_value: float, degrees_of_freedom: float, confidence_interval: array{float, float}} $expected */
            $expected = $entry[$key];

            if ($expected['statistic'] === 0.0) {
                self::assertSame(0.0, $test->statistic, "{$name}: {$key} statistic");
                self::assertSame(1.0, $test->pValue(), "{$name}: {$key} p-value");
            } else {
                Lre::assertDigits($test->statistic, $expected['statistic'], "{$name}: {$key} t", 12.0);
                Lre::assertDigits($test->pValue(), $expected['p_value'], "{$name}: {$key} p", 11.0);
            }

            Lre::assertDigits(
                $test->degreesOfFreedom,
                $expected['degrees_of_freedom'],
                "{$name}: {$key} degrees of freedom",
                12.0,
            );

            [$low, $high] = $test->confidenceInterval();
            Lre::assertDigits($low, $expected['confidence_interval'][0], "{$name}: {$key} interval lower", 11.0);
            Lre::assertDigits($high, $expected['confidence_interval'][1], "{$name}: {$key} interval upper", 11.0);
        }
    }

    /**
     * Welch and Student agree exactly when the samples are the same size, and
     * Welch has the fewer degrees of freedom otherwise.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('samples')]
    public function test_welch_never_claims_more_degrees_of_freedom_than_student(
        string $name,
        array $entry,
    ): void {
        /** @var array{x: list<float>, y: list<float>} $entry */
        $student = TTest::student($entry['x'], $entry['y']);
        $welch = TTest::welch($entry['x'], $entry['y']);

        // The tolerance goes on the Student side: the two are exactly equal
        // whenever the samples are the same size and the variances match.
        self::assertLessThanOrEqual($student->degreesOfFreedom + 1.0e-9, $welch->degreesOfFreedom, $name);
    }

    /**
     * Which variant produced the result is recorded on it, so a number that
     * has been passed around can still say what it assumed.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('samples')]
    public function test_each_variant_says_which_one_it_is(string $name, array $entry): void
    {
        /** @var array{x: list<float>, y: list<float>} $entry */
        self::assertTrue(TTest::student($entry['x'], $entry['y'])->pooled, "{$name}: student pools");
        self::assertFalse(TTest::welch($entry['x'], $entry['y'])->pooled, "{$name}: welch does not");
    }

    /**
     * An exactly determined difference is infinitely many standard errors from
     * zero, not an error -- and when the difference is zero as well it is
     * undefined rather than infinite. Fit::tStatistic() takes the same
     * position, and this is the case that reaches it: two constant samples.
     */
    public function test_a_zero_standard_error_is_infinite_rather_than_an_error(): void
    {
        $apart = TTest::welch([5.0, 5.0, 5.0], [3.0, 3.0, 3.0]);

        self::assertSame(INF, $apart->statistic);
        self::assertSame(0.0, $apart->pValue());
        self::assertSame(0.0, $apart->standardError);

        $identical = TTest::welch([5.0, 5.0, 5.0], [5.0, 5.0, 5.0]);

        self::assertNan($identical->statistic, 'zero over zero is undefined, not infinite');
        self::assertNan($identical->pValue());

        // The Satterthwaite denominator vanishes with both variances, and
        // falling back on the pooled degrees of freedom keeps the object
        // constructible rather than dividing by zero to build it.
        self::assertSame(4.0, $identical->degreesOfFreedom);
    }

    /** @param array<string, mixed> $entry */
    #[DataProvider('rankSamples')]
    public function test_the_mann_whitney_statistic_matches_scipy(string $name, array $entry): void
    {
        /** @var array<string, mixed> $entry */
        /** @var array{x: list<float>, y: list<float>} $sample */
        $sample = $entry;

        $alternatives = [
            'two-sided' => Alternative::TwoSided,
            'less' => Alternative::Less,
            'greater' => Alternative::Greater,
        ];

        foreach ($alternatives as $label => $alternative) {
            foreach (['corrected' => Continuity::Corrected, 'uncorrected' => Continuity::Uncorrected] as $key => $mode) {
                /** @var array{statistic: float, p_value: float} $expected */
                $expected = $entry["{$label}_{$key}"];

                $test = MannWhitneyU::of($sample['x'], $sample['y'], $alternative, $mode);

                Lre::assertDigits(
                    $test->statistic,
                    $expected['statistic'],
                    "{$name}: U for {$label}/{$key}",
                    14.0,
                );
                Lre::assertDigits(
                    $test->pValue(),
                    $expected['p_value'],
                    "{$name}: p for {$label}/{$key}",
                    11.0,
                );
            }
        }
    }

    /**
     * The two one-sided p-values sum to one plus the shared boundary mass,
     * which without the continuity correction is exactly one.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('rankSamples')]
    public function test_the_one_sided_tails_are_complementary(string $name, array $entry): void
    {
        /** @var array{x: list<float>, y: list<float>} $entry */
        $less = MannWhitneyU::of($entry['x'], $entry['y'], Alternative::Less, Continuity::Uncorrected);
        $greater = MannWhitneyU::of($entry['x'], $entry['y'], Alternative::Greater, Continuity::Uncorrected);

        self::assertEqualsWithDelta(1.0, $less->pValue() + $greater->pValue(), 1.0e-12, $name);
    }

    /** @param array<string, mixed> $entry */
    #[DataProvider('groupSets')]
    public function test_the_kruskal_wallis_statistic_matches_scipy(string $name, array $entry): void
    {
        /** @var array{groups: list<list<float>>, statistic: float, p_value: float, degrees_of_freedom: int} $entry */
        $test = KruskalWallis::of($entry['groups']);

        self::assertSame($entry['degrees_of_freedom'], $test->degreesOfFreedom, "{$name}: df");

        if ($entry['statistic'] === 0.0) {
            self::assertEqualsWithDelta(0.0, $test->statistic, 1.0e-12, "{$name}: H");
            self::assertSame(1.0, $test->pValue(), "{$name}: p");
        } else {
            Lre::assertDigits($test->statistic, $entry['statistic'], "{$name}: H", 12.0);
            Lre::assertDigits($test->pValue(), $entry['p_value'], "{$name}: p", 11.0);
        }
    }

    /**
     * The tie correction is a divisor, so it can only raise the statistic, and
     * it is exactly one when nothing is tied.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('groupSets')]
    public function test_the_tie_correction_reads_as_a_divisor(string $name, array $entry): void
    {
        /** @var array{groups: list<list<float>>} $entry */
        $test = KruskalWallis::of($entry['groups']);

        self::assertEqualsWithDelta(
            $test->statistic,
            $test->uncorrectedStatistic / $test->tieCorrection,
            1.0e-9,
            "{$name}: the corrected statistic is not the uncorrected one over the factor",
        );

        self::assertLessThanOrEqual(1.0, $test->tieCorrection, $name);
        self::assertGreaterThan(0.0, $test->tieCorrection, $name);
    }

    /**
     * An empty group is dropped, not refused and not counted.
     *
     * OneWayAnova::of() does the same, and the two should be swappable at the
     * call site. Nothing covered it: mutation testing removed the filter and
     * every test still passed.
     */
    public function test_an_empty_group_is_dropped(): void
    {
        $without = KruskalWallis::of([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        $with = KruskalWallis::of([[1.0, 2.0, 3.0], [], [4.0, 5.0, 6.0]]);

        self::assertSame($without->groups, $with->groups, 'the empty group was counted');
        self::assertSame($without->degreesOfFreedom, $with->degreesOfFreedom);
        self::assertSame($without->statistic, $with->statistic);
        self::assertSame($without->observations, $with->observations);
    }

    /**
     * Grouped input describes the same analysis as separated input.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('groupSets')]
    public function test_grouped_input_agrees_with_separated_input(string $name, array $entry): void
    {
        /** @var array{groups: list<list<float>>} $entry */
        $values = [];
        $labels = [];

        foreach ($entry['groups'] as $index => $group) {
            foreach ($group as $value) {
                $values[] = $value;
                $labels[] = $index;
            }
        }

        self::assertEqualsWithDelta(
            KruskalWallis::of($entry['groups'])->statistic,
            KruskalWallis::grouped($values, $labels)->statistic,
            1.0e-12,
            $name,
        );
    }
}
