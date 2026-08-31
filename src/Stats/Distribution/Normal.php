<?php

declare(strict_types=1);

namespace Vegoia\Stats\Distribution;

use function exp;
use function log;
use function sqrt;

use Vegoia\Stats\SpecialFunction;

/**
 * The normal distribution.
 *
 * The quantile does not go through the iteration in the base class. Wichura's
 * AS 241 gives it directly to about sixteen digits through two rational
 * functions and a third branch for the far tail, and it is the routine R,
 * Boost and the GSL all use. Newton would reach the same place more slowly
 * and only if the bracket search had already found the region -- which, at
 * p = 1e-300 where the answer is 37.0, is the part that is actually hard.
 */
final class Normal extends ContinuousDistribution
{
    private const float SQRT_TWO = 1.4142135623730951;

    private const float SQRT_TWO_PI = 2.5066282746310002;

    public function __construct(
        public readonly float $mean = 0.0,
        public readonly float $standardDeviation = 1.0,
    ) {
        self::assertPositiveShape('the standard deviation', $standardDeviation);
    }

    public function density(float $x): float
    {
        $z = ($x - $this->mean) / $this->standardDeviation;

        return exp(-0.5 * $z * $z) / ($this->standardDeviation * self::SQRT_TWO_PI);
    }

    public function cumulative(float $x): float
    {
        // erfc rather than 1 + erf: below the mean this is the small tail, and
        // erfc is the routine that keeps its digits there.
        return 0.5 * SpecialFunction::erfc(-$this->standardise($x) / self::SQRT_TWO);
    }

    public function survival(float $x): float
    {
        return 0.5 * SpecialFunction::erfc($this->standardise($x) / self::SQRT_TWO);
    }

    public function quantile(float $p): float
    {
        self::assertProbability($p);

        return $this->mean + $this->standardDeviation * self::standardQuantile($p);
    }

    public function upperQuantile(float $p): float
    {
        self::assertProbability($p);

        // The normal is symmetric, so the upper quantile is the lower one
        // reflected -- and reflecting is exact, where computing 1 - p is not.
        return $this->mean - $this->standardDeviation * self::standardQuantile($p);
    }

    /**
     * Wichura's AS 241 (PPND16), the inverse of the standard normal cumulative.
     *
     * Three regions, each with its own rational approximation: the central
     * one in terms of q^2 where q = p - 1/2, and two tail branches in terms
     * of sqrt(-log(min(p, 1-p))). The third branch is what lets the far tail
     * work at all -- it holds to r = 27, which is p near 1e-316, so the
     * 1e-300 quantile is inside its range rather than past the end of it.
     */
    private static function standardQuantile(float $p): float
    {
        if ($p <= 0.0) {
            return -INF;
        }

        if ($p >= 1.0) {
            return INF;
        }

        $q = $p - 0.5;

        if (abs($q) <= 0.425) {
            $r = 0.180625 - $q * $q;

            return $q * self::polynomial($r, [
                2509.0809287301226727, 33430.575583588128105, 67265.770927008700853,
                45921.953931549871457, 13731.693765509461125, 1971.5909503065514427,
                133.14166789178437745, 3.387132872796366608,
            ]) / self::polynomial($r, [
                5226.495278852854561, 28729.085735721942674, 39307.89580009271061,
                21213.794301586595867, 5394.1960214247511077, 687.1870074920579083,
                42.313330701600911252, 1.0,
            ]);
        }

        $r = $q < 0.0 ? $p : 1.0 - $p;
        $r = sqrt(-log($r));

        if ($r <= 5.0) {
            $r -= 1.6;
            $value = self::polynomial($r, [
                7.7454501427834140764e-4, 0.0227238449892691845833, 0.24178072517745061177,
                1.27045825245236838258, 3.64784832476320460504, 5.7694972214606914055,
                4.6303378461565452959, 1.42343711074968357734,
            ]) / self::polynomial($r, [
                1.05075007164441684324e-9, 5.475938084995344946e-4, 0.0151986665636164571966,
                0.14810397642748007459, 0.68976733498510000455, 1.6763848301838038494,
                2.05319162663775882187, 1.0,
            ]);
        } else {
            $r -= 5.0;
            $value = self::polynomial($r, [
                2.01033439929228813265e-7, 2.71155556874348757815e-5, 0.0012426609473880784386,
                0.026532189526576123093, 0.29656057182850489123, 1.7848265399172913358,
                5.4637849111641143699, 6.6579046435011037772,
            ]) / self::polynomial($r, [
                2.04426310338993978564e-15, 1.4215117583164458887e-7, 1.8463183175100546818e-5,
                7.868691311456132591e-4, 0.0148753612908506148525, 0.13692988092273580531,
                0.59983220655588793769, 1.0,
            ]);
        }

        return $q < 0.0 ? -$value : $value;
    }

    /**
     * Horner from the constant term upwards.
     *
     * @param list<float> $coefficients highest power first
     */
    private static function polynomial(float $x, array $coefficients): float
    {
        $value = 0.0;

        foreach ($coefficients as $coefficient) {
            $value = $value * $x + $coefficient;
        }

        return $value;
    }

    private function standardise(float $x): float
    {
        return ($x - $this->mean) / $this->standardDeviation;
    }

    protected function guessUpperQuantile(float $p): float
    {
        return $this->upperQuantile($p);
    }

    protected function infimum(): float
    {
        return -INF;
    }
}
