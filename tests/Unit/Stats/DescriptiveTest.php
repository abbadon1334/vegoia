<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit\Stats;

use ArrayIterator;

use function file_get_contents;
use function json_decode;

use const JSON_THROW_ON_ERROR;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Stats\Descriptive;
use Vegoia\Tests\Support\Lre;
use Vegoia\Tests\Support\Paths;

#[CoversClass(Descriptive::class)]
final class DescriptiveTest extends TestCase
{
    /** @var list<float> */
    private const array SAMPLE = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

    public function test_the_textbook_example(): void
    {
        $stats = Descriptive::of(self::SAMPLE);

        self::assertSame(8, $stats->count());
        self::assertSame(40.0, $stats->sum());
        self::assertSame(5.0, $stats->mean());
        self::assertSame(2.0, $stats->min());
        self::assertSame(9.0, $stats->max());
        self::assertSame(7.0, $stats->range());
        self::assertSame(4.0, $stats->populationVariance());
        self::assertSame(2.0, $stats->populationStdDev());
        self::assertEqualsWithDelta(4.5714285714285714, $stats->variance(), 1.0e-15);
    }

    /**
     * Quantiles have nine competing definitions that disagree on the same data.
     * This is R's type 7, the default in R and numpy, stated explicitly so a
     * result can be reconciled with theirs rather than silently differing.
     */
    public function test_quantiles_follow_the_r_type_seven_definition(): void
    {
        $stats = Descriptive::of([1.0, 2.0, 3.0, 4.0]);

        self::assertSame(1.0, $stats->quantile(0.0));
        self::assertSame(1.75, $stats->quantile(0.25));
        self::assertSame(2.5, $stats->quantile(0.5));
        self::assertSame(2.5, $stats->median());
        self::assertSame(3.25, $stats->quantile(0.75));
        self::assertSame(4.0, $stats->quantile(1.0));
        self::assertSame(1.5, $stats->iqr());
    }

    public function test_the_median_of_an_odd_sample_is_the_middle_value(): void
    {
        self::assertSame(3.0, Descriptive::of([5.0, 1.0, 3.0])->median());
    }

    public function test_quantiles_do_not_depend_on_input_order(): void
    {
        self::assertSame(
            Descriptive::of([3.0, 1.0, 4.0, 1.0, 5.0])->median(),
            Descriptive::of([5.0, 4.0, 3.0, 1.0, 1.0])->median(),
        );
    }

    public function test_a_symmetric_sample_has_no_skew(): void
    {
        self::assertEqualsWithDelta(0.0, Descriptive::of([1.0, 2.0, 3.0, 4.0, 5.0])->skewness(), 1.0e-15);
    }

    public function test_skewness_has_the_sign_of_the_longer_tail(): void
    {
        self::assertGreaterThan(0.0, Descriptive::of([1.0, 1.0, 1.0, 2.0, 10.0])->skewness());
        self::assertLessThan(0.0, Descriptive::of([-10.0, -2.0, 1.0, 1.0, 1.0])->skewness());
    }

    /** Excess kurtosis: zero for a normal distribution, not three. */
    public function test_kurtosis_is_reported_as_excess(): void
    {
        // A two-point distribution is maximally flat: excess kurtosis -2.
        self::assertEqualsWithDelta(-2.0, Descriptive::of([0.0, 1.0, 0.0, 1.0])->kurtosis(), 1.0e-12);
    }

    public function test_standard_error_shrinks_with_the_square_root_of_n(): void
    {
        $stats = Descriptive::of(self::SAMPLE);

        self::assertEqualsWithDelta($stats->stdDev() / sqrt(8.0), $stats->standardError(), 1.0e-15);
    }

    /**
     * Built on the sample standard deviation, matching `stdDev()` and R's
     * `sd(x) / mean(x)`. The population form would give 0.4 here, so the choice
     * is visible rather than assumed.
     */
    public function test_coefficient_of_variation_is_relative_dispersion(): void
    {
        $stats = Descriptive::of(self::SAMPLE);

        self::assertSame($stats->stdDev() / $stats->mean(), $stats->coefficientOfVariation());
        self::assertEqualsWithDelta(0.4276179870, $stats->coefficientOfVariation(), 1.0e-9);
    }

    /**
     * The residual correction in the two-pass variance, on a sample large
     * enough to need it.
     *
     * Mutation testing found this gap: deleting the correction left every
     * existing test green, because the NIST univariate sets top out at 5000
     * observations and the residual sum(x - mean) stays negligible at that
     * size once the mean itself is accumulated with compensation. At 200000
     * values clustered within a micro-unit of 1e9 it is not negligible --
     * dropping the correction moves the variance in its fourth significant
     * digit.
     *
     * The expected values come from `tools/generate_variance_stress.py`, which
     * computes them in exact rational arithmetic, so the target carries no
     * rounding of its own.
     */
    public function test_the_residual_correction_matters_on_a_large_tight_sample(): void
    {
        /** @var array{count: int, base: int, mean: float, variance: float, standard_deviation: float} $spec */
        $spec = json_decode(
            (string) file_get_contents(Paths::fixture('variance_stress.json')),
            associative: true,
            flags: JSON_THROW_ON_ERROR,
        );

        $values = [];
        for ($i = 0; $i < $spec['count']; $i++) {
            $values[] = $spec['base'] + ($i % 7) * 1.0e-6;
        }

        $stats = Descriptive::of($values);

        Lre::assertDigits($stats->mean(), $spec['mean'], 'mean of the stress sample', digits: 14);
        Lre::assertDigits($stats->variance(), $spec['variance'], 'variance of the stress sample', digits: 9);
        Lre::assertDigits(
            $stats->stdDev(),
            $spec['standard_deviation'],
            'standard deviation of the stress sample',
            digits: 9,
        );
    }

    public function test_a_constant_sample_has_zero_spread(): void
    {
        $stats = Descriptive::of([3.0, 3.0, 3.0, 3.0]);

        self::assertSame(3.0, $stats->mean());
        self::assertSame(0.0, $stats->variance());
        self::assertSame(0.0, $stats->stdDev());
    }

    public function test_it_accepts_any_iterable_and_non_list_arrays(): void
    {
        self::assertSame(2.0, Descriptive::of(['a' => 1.0, 'b' => 3.0])->mean());
        self::assertSame(2.0, Descriptive::of(new ArrayIterator([1, 3]))->mean());
    }

    /** @return iterable<string, array{callable(Descriptive): mixed, list<float>}> */
    public static function refusals(): iterable
    {
        yield 'mean of nothing' => [static fn (Descriptive $d) => $d->mean(), []];
        yield 'min of nothing' => [static fn (Descriptive $d) => $d->min(), []];
        yield 'max of nothing' => [static fn (Descriptive $d) => $d->max(), []];
        yield 'quantile of nothing' => [static fn (Descriptive $d) => $d->median(), []];
        yield 'variance of one value' => [static fn (Descriptive $d) => $d->variance(), [1.0]];
        yield 'skewness of two values' => [static fn (Descriptive $d) => $d->skewness(), [1.0, 2.0]];
        yield 'kurtosis of three values' => [static fn (Descriptive $d) => $d->kurtosis(), [1.0, 2.0, 3.0]];
        yield 'autocorrelation past the end' => [static fn (Descriptive $d) => $d->autocorrelation(5), [1.0, 2.0]];
        yield 'autocorrelation at lag zero' => [static fn (Descriptive $d) => $d->autocorrelation(0), [1.0, 2.0]];
    }

    /**
     * @param callable(Descriptive): mixed $operation
     * @param list<float>                  $values
     */
    #[DataProvider('refusals')]
    public function test_it_refuses_what_it_cannot_compute(callable $operation, array $values): void
    {
        $this->expectException(InvalidArgument::class);

        $operation(Descriptive::of($values));
    }

    public function test_a_quantile_probability_outside_zero_to_one_is_refused(): void
    {
        $this->expectException(InvalidArgument::class);

        Descriptive::of(self::SAMPLE)->quantile(1.5);
    }
}
