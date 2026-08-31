<?php

declare(strict_types=1);

namespace Vegoia\Stats\Distribution;

use Vegoia\Stats\SpecialFunction;

/**
 * The F distribution: the ratio of two variances, and so the p-value of an
 * analysis of variance or of a nested model comparison.
 *
 * The two tails use the two orderings of the same incomplete beta, which is
 * how each is computed as a small number rather than as one minus a large one:
 *
 *     P(F <= x) = I_z(d1/2, d2/2)      with z = d1 x / (d1 x + d2)
 *     P(F >  x) = I_{1-z}(d2/2, d1/2)
 *
 * 1 - z is formed as d2 / (d1 x + d2) rather than by subtracting, because at
 * large x the subtraction has nothing left: at x = 1000 with (10, 100) degrees
 * of freedom, z is 0.99 and the tail is 1e-49.
 */
final class FisherSnedecor extends ContinuousDistribution
{
    public function __construct(
        public readonly float $numeratorDegreesOfFreedom,
        public readonly float $denominatorDegreesOfFreedom,
    ) {
        self::assertPositiveShape('the numerator degrees of freedom', $numeratorDegreesOfFreedom);
        self::assertPositiveShape('the denominator degrees of freedom', $denominatorDegreesOfFreedom);
    }

    public function density(float $x): float
    {
        if ($x <= 0.0) {
            return $x < 0.0 || $this->numeratorDegreesOfFreedom > 2.0 ? 0.0 : INF;
        }

        // z^a (1-z)^b / B(a, b), over x. Taken from SpecialFunction rather
        // than assembled here from logGamma: that form subtracts 165.6 from
        // 147.7 to reach -17.9 at (100, 10) degrees of freedom, and measured
        // 12.80 digits against SciPy's 13.33.
        return SpecialFunction::betaPrefactor(
            $this->numeratorShare($x),
            $this->numeratorDegreesOfFreedom / 2.0,
            $this->denominatorDegreesOfFreedom / 2.0,
            $this->denominatorShare($x),
        ) / $x;
    }

    public function cumulative(float $x): float
    {
        if ($x <= 0.0) {
            return 0.0;
        }

        return SpecialFunction::regularizedBeta(
            $this->numeratorShare($x),
            $this->numeratorDegreesOfFreedom / 2.0,
            $this->denominatorDegreesOfFreedom / 2.0,
            $this->denominatorShare($x),
        );
    }

    public function survival(float $x): float
    {
        if ($x <= 0.0) {
            return 1.0;
        }

        return SpecialFunction::regularizedBeta(
            $this->denominatorShare($x),
            $this->denominatorDegreesOfFreedom / 2.0,
            $this->numeratorDegreesOfFreedom / 2.0,
            $this->numeratorShare($x),
        );
    }

    protected function guessUpperQuantile(float $p): float
    {
        $d1 = $this->numeratorDegreesOfFreedom;
        $d2 = $this->denominatorDegreesOfFreedom;

        // An F is a ratio of two chi-squareds over their degrees of freedom,
        // so the upper quantile of the numerator over the median of the
        // denominator is in the right region even when it is not close.
        $upper = new ChiSquared($d1)->upperQuantile($p) / $d1;
        $lower = new ChiSquared($d2)->upperQuantile(0.5) / $d2;

        return $lower > 0.0 ? $upper / $lower : $upper;
    }

    protected function infimum(): float
    {
        return 0.0;
    }

    /** z = d1 x / (d1 x + d2), the beta argument for the lower tail. */
    private function numeratorShare(float $x): float
    {
        $scaled = $this->numeratorDegreesOfFreedom * $x;

        return $scaled / ($scaled + $this->denominatorDegreesOfFreedom);
    }

    /** 1 - z, formed directly so the far tail is not a subtraction. */
    private function denominatorShare(float $x): float
    {
        $scaled = $this->numeratorDegreesOfFreedom * $x;

        return $this->denominatorDegreesOfFreedom / ($scaled + $this->denominatorDegreesOfFreedom);
    }
}
