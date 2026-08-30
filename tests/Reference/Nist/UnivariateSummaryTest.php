<?php

declare(strict_types=1);

namespace Vegoia\Tests\Reference\Nist;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Vegoia\Stats\Descriptive;
use Vegoia\Tests\Support\AttainableAccuracy;
use Vegoia\Tests\Support\Lre;
use Vegoia\Tests\Support\NistUnivariate;

/**
 * Summary statistics against the NIST Statistical Reference Datasets.
 *
 * These are not smoke tests. The four NumAcc sets are constructed so that the
 * textbook one-pass variance, sum(x^2)/n - mean^2, loses every significant
 * digit to catastrophic cancellation: NumAcc1 holds three 8-digit integers
 * whose standard deviation is exactly 1, and the naive formula returns 0.
 * Passing this file is evidence the implementation is numerically sound, not
 * merely that it compiles.
 *
 * @see https://www.itl.nist.gov/div898/strd/univ/homepage.html
 */
#[CoversClass(Descriptive::class)]
#[Group('reference')]
#[Group('nist')]
final class UnivariateSummaryTest extends TestCase
{
    /** @return iterable<string, array{string}> */
    public static function datasets(): iterable
    {
        // Ordered from benign to hostile, so a failure list reads as a diagnosis.
        foreach (['PiDigits', 'Lottery', 'Lew', 'Mavro', 'Michelso',
                  'NumAcc1', 'NumAcc2', 'NumAcc3', 'NumAcc4'] as $name) {
            yield $name => [$name];
        }
    }

    #[DataProvider('datasets')]
    public function test_mean_matches_certified_value(string $name): void
    {
        $set = NistUnivariate::load($name);

        Lre::assertDigits(
            Descriptive::of($set->values)->mean(),
            $set->mean,
            "{$name}: sample mean",
            AttainableAccuracy::required($name, 'mean'),
        );
    }

    #[DataProvider('datasets')]
    public function test_standard_deviation_matches_certified_value(string $name): void
    {
        $set = NistUnivariate::load($name);

        Lre::assertDigits(
            Descriptive::of($set->values)->stdDev(),
            $set->stdDev,
            "{$name}: sample standard deviation (denom. n-1)",
            AttainableAccuracy::required($name, 'stdDev'),
        );
    }

    #[DataProvider('datasets')]
    public function test_lag_one_autocorrelation_matches_certified_value(string $name): void
    {
        $set = NistUnivariate::load($name);

        Lre::assertDigits(
            Descriptive::of($set->values)->autocorrelation(1),
            $set->autocorrelation,
            "{$name}: lag-1 autocorrelation",
            AttainableAccuracy::required($name, 'autocorrelation'),
        );
    }

    public function test_the_fixture_parser_reads_what_the_file_declares(): void
    {
        // Guards the parser itself: if this drifts, every assertion above
        // becomes meaningless while still reporting green.
        $numAcc1 = NistUnivariate::load('NumAcc1');

        self::assertSame([10000001.0, 10000003.0, 10000002.0], $numAcc1->values);
        self::assertSame(10000002.0, $numAcc1->mean);
        self::assertSame(1.0, $numAcc1->stdDev);
        self::assertSame(-0.5, $numAcc1->autocorrelation);

        $piDigits = NistUnivariate::load('PiDigits');

        self::assertCount(5000, $piDigits->values);
        self::assertSame(4.5348, $piDigits->mean);
    }
}
