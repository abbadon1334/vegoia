<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit\Stats;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;
use Vegoia\Stats\Descriptive;
use Vegoia\Stats\Precision;
use Vegoia\Tests\Support\Lre;
use Vegoia\Tests\Support\NistUnivariate;

/**
 * The two precision modes, and the boundary between them.
 *
 * Offering a faster, less accurate path is a real risk: the fast answer looks
 * exactly like the accurate one, so a wrong default would be invisible. These
 * tests pin three things -- that Extended is the default, that Fast agrees
 * with it on well-conditioned data, and that on the data NIST built to expose
 * the difference it does not.
 *
 * That last one matters most. If Fast quietly matched Extended everywhere, the
 * extra cost would be waste and should be removed; the reason to keep both is
 * that the gap is real and measurable.
 */
#[CoversClass(Precision::class)]
#[CoversClass(Descriptive::class)]
final class PrecisionTest extends TestCase
{
    public function test_extended_is_the_default(): void
    {
        self::assertSame(Precision::Extended, Descriptive::of([1.0, 2.0])->precision());
        self::assertSame(Precision::Fast, Descriptive::of([1.0, 2.0], Precision::Fast)->precision());
    }

    public function test_with_returns_the_same_sample_computed_the_other_way(): void
    {
        $extended = Descriptive::of([1.0, 2.0, 3.0]);
        $fast = $extended->with(Precision::Fast);

        self::assertSame(Precision::Fast, $fast->precision());
        self::assertSame($extended->values(), $fast->values());
        self::assertSame($extended, $extended->with(Precision::Extended), 'no copy when nothing changes');
    }

    /**
     * On ordinary data the two agree closely -- Fast is not a different
     * statistic, just the same one computed without compensation.
     *
     * @return iterable<string, array{string}>
     */
    public static function wellConditioned(): iterable
    {
        foreach (['PiDigits', 'Lottery', 'Lew', 'Michelso'] as $name) {
            yield $name => [$name];
        }
    }

    #[DataProvider('wellConditioned')]
    public function test_fast_stays_close_to_extended_on_ordinary_data(string $name): void
    {
        $set = NistUnivariate::load($name);

        $extended = Descriptive::of($set->values);
        $fast = Descriptive::of($set->values, Precision::Fast);

        // Both certified to at least twelve digits; the gap between them is
        // smaller than the gap either has to the certified value.
        Lre::assertDigits($fast->mean(), $set->mean, "{$name}: fast mean", digits: 12);
        Lre::assertDigits($fast->stdDev(), $set->stdDev, "{$name}: fast stdDev", digits: 12);

        Lre::assertDigits($fast->mean(), $extended->mean(), "{$name}: fast vs extended mean", digits: 12);
    }

    /**
     * And on the datasets built to break naive arithmetic, it does not -- which
     * is the entire reason Extended exists and is the default.
     *
     * NumAcc4 holds 8-digit values differing in their last places. Plain
     * summation reaches 9 correct digits on its lag-1 autocorrelation;
     * compensated summation with exact products reaches 15.65.
     */
    public function test_the_modes_diverge_where_naive_arithmetic_fails(): void
    {
        $set = NistUnivariate::load('NumAcc4');

        $fastError = abs(Descriptive::of($set->values, Precision::Fast)->autocorrelation(1) - $set->autocorrelation);
        $exactError = abs(Descriptive::of($set->values)->autocorrelation(1) - $set->autocorrelation);

        self::assertGreaterThan(
            $exactError * 1000,
            $fastError,
            'extended must be orders of magnitude closer here, or it is not earning its cost',
        );
    }

    /** Every statistic must be available in both modes, not just the tested ones. */
    public function test_both_modes_answer_the_whole_interface(): void
    {
        $values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

        foreach ([Precision::Fast, Precision::Extended] as $precision) {
            $stats = Descriptive::of($values, $precision);

            self::assertEqualsWithDelta(5.0, $stats->mean(), 1.0e-12, $precision->name);
            self::assertEqualsWithDelta(4.0, $stats->populationVariance(), 1.0e-12, $precision->name);
            self::assertEqualsWithDelta(2.0, $stats->populationStdDev(), 1.0e-12, $precision->name);
            // Eight values: the median is the mean of the two middle ones.
            self::assertSame(4.5, $stats->median(), $precision->name);
            self::assertEqualsWithDelta(0.0, $stats->skewness(), 1.0, $precision->name);
        }
    }
}
