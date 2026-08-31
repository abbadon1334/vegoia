<?php

declare(strict_types=1);

namespace Vegoia\Stats\Distribution;

use function sqrt;

use Vegoia\Stats\SpecialFunction;

/**
 * The chi-squared distribution: goodness of fit, likelihood ratios, variance.
 *
 * Both tails are the incomplete gamma at a = k/2, x/2, taken from the two
 * routines that compute each one directly rather than by complementing the
 * other -- which is what lets the upper tail stay meaningful at 500 degrees of
 * freedom and x = 5000, where it is around 1e-300.
 */
final class ChiSquared extends ContinuousDistribution
{
    public function __construct(public readonly float $degreesOfFreedom)
    {
        self::assertPositiveShape('the degrees of freedom', $degreesOfFreedom);
    }

    public function density(float $x): float
    {
        if ($x < 0.0) {
            return 0.0;
        }

        if ($x === 0.0) {
            // The density is finite at zero only for two degrees of freedom,
            // infinite below that and zero above; taking the logarithm would
            // report none of the three.
            return match (true) {
                $this->degreesOfFreedom < 2.0 => INF,
                $this->degreesOfFreedom === 2.0 => 0.5,
                default => 0.0,
            };
        }

        // (x/2)^(k/2) e^(-x/2) / gamma(k/2), over x -- which is the chi-squared
        // density rearranged so the whole prefactor comes from the compensated
        // routine rather than from a logGamma of 1128 at 500 degrees of freedom.
        return SpecialFunction::gammaPrefactor($this->degreesOfFreedom / 2.0, $x / 2.0) / $x;
    }

    public function cumulative(float $x): float
    {
        return $x <= 0.0 ? 0.0 : SpecialFunction::regularizedGammaP($this->degreesOfFreedom / 2.0, $x / 2.0);
    }

    public function survival(float $x): float
    {
        return $x <= 0.0 ? 1.0 : SpecialFunction::regularizedGammaQ($this->degreesOfFreedom / 2.0, $x / 2.0);
    }

    protected function guessUpperQuantile(float $p): float
    {
        $k = $this->degreesOfFreedom;

        // Wilson-Hilferty: the cube root of a chi-squared over its degrees of
        // freedom is very nearly normal, which turns the normal quantile into
        // a chi-squared one in closed form.
        $z = new Normal()->upperQuantile($p);
        $term = 1.0 - 2.0 / (9.0 * $k) + $z * sqrt(2.0 / (9.0 * $k));

        return $term > 0.0 ? $k * $term ** 3 : $k / 100.0;
    }

    protected function infimum(): float
    {
        return 0.0;
    }
}
