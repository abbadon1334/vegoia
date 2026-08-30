<?php

declare(strict_types=1);

namespace Vegoia\Tests\Support;

use function abs;
use function is_finite;
use function log10;

use PHPUnit\Framework\Assert;

use function sprintf;

/**
 * Accuracy measured the way NIST measures it.
 *
 * A plain `assertEqualsWithDelta` forces you to invent an absolute epsilon,
 * which is meaningless when certified values range from 0.0004 to 4.2e6. The
 * literature around the Statistical Reference Datasets instead reports the
 * Log Relative Error -- roughly "how many correct significant digits did you
 * get" -- so a single threshold is comparable across every dataset.
 *
 *     LRE = -log10(|computed - certified| / |certified|)
 *
 * An IEEE-754 double carries about 15.95 decimal digits, so LRE 15+ means the
 * answer is as good as the format allows. NIST considers a package adequate
 * from about 11; a naive one-pass variance drops to 0 (or negative) on the
 * NumAcc datasets, which is precisely what those datasets exist to expose.
 */
final class Lre
{
    /** Digits we require by default: strict enough to reject a naive algorithm. */
    public const int DEFAULT_DIGITS = 13;

    public static function of(float $computed, float $certified): float
    {
        if ($computed === $certified) {
            return INF;
        }

        // A certified zero has no relative scale, so fall back to absolute error.
        $denominator = $certified === 0.0 ? 1.0 : abs($certified);

        $error = abs($computed - $certified) / $denominator;

        return $error === 0.0 ? INF : -log10($error);
    }

    public static function assertDigits(
        float $computed,
        float $certified,
        string $what,
        float $digits = self::DEFAULT_DIGITS,
    ): void {
        $lre = self::of($computed, $certified);

        Assert::assertGreaterThanOrEqual(
            $digits,
            $lre,
            sprintf(
                "%s is accurate to only %s significant digits (required %.2f).\n"
                . "  certified : %.17g\n  computed  : %.17g\n  abs error : %.6g",
                $what,
                is_finite($lre) ? sprintf('%.2f', $lre) : 'inf',
                $digits,
                $certified,
                $computed,
                abs($computed - $certified),
            ),
        );
    }
}
