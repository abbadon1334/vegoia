<?php

declare(strict_types=1);

namespace Vegoia\Stats\Distribution;

use function exp;
use function log;
use function log1p;

use Vegoia\Stats\SpecialFunction;

/**
 * Student's t distribution, the one a regression coefficient is tested against.
 *
 * Both tails come from the same incomplete beta, evaluated at
 * k / (k + t^2), which is small when |t| is large. That argument is the reason
 * the far tail survives: it approaches zero rather than one as t grows, so the
 * quantity being computed is the small one and no cancellation is involved.
 *
 * The tails are heavy, and heavy in a way that matters for the inverse. With
 * one degree of freedom this is the Cauchy distribution, whose 1e-12 upper
 * quantile is 3.2e11 -- eleven orders of magnitude further out than the
 * normal's 7.0. A guess built from the normal is therefore useless on its own
 * here, and the guess below uses the power-law form of the tail instead when
 * the degrees of freedom are few.
 */
final class StudentT extends ContinuousDistribution
{
    private readonly float $logDensityConstant;

    public function __construct(public readonly float $degreesOfFreedom)
    {
        self::assertPositiveShape('the degrees of freedom', $degreesOfFreedom);

        $k = $degreesOfFreedom;
        $this->logDensityConstant = SpecialFunction::logGamma(($k + 1.0) / 2.0)
            - SpecialFunction::logGamma($k / 2.0)
            - 0.5 * log($k * M_PI);
    }

    public function density(float $x): float
    {
        $k = $this->degreesOfFreedom;

        // log1p rather than log(1 + t^2/k): at large t the ratio dominates and
        // this makes no difference, but at small t it is the whole value.
        return exp($this->logDensityConstant - ($k + 1.0) / 2.0 * log1p($x * $x / $k));
    }

    public function cumulative(float $x): float
    {
        return $x >= 0.0 ? 1.0 - $this->upperTail($x) : $this->upperTail(-$x);
    }

    public function survival(float $x): float
    {
        return $x >= 0.0 ? $this->upperTail($x) : 1.0 - $this->upperTail(-$x);
    }

    public function upperQuantile(float $p): float
    {
        self::assertProbability($p);

        if ($p === 0.5) {
            return 0.0;
        }

        // Symmetry, applied before the search: reflecting is exact where
        // computing 1 - p is not, and it keeps the bracket on the half where
        // the origin at zero is a valid starting point.
        return $p > 0.5
            ? -parent::upperQuantile(1.0 - $p)
            : parent::upperQuantile($p);
    }

    /** P(T > t) for t >= 0, which is the tail that keeps its digits. */
    private function upperTail(float $t): float
    {
        $k = $this->degreesOfFreedom;

        // Both k/(k+t^2) and its complement t^2/(k+t^2) are formed directly;
        // subtracting one from the other loses the small one when t is large.
        $scale = $k + $t * $t;

        return 0.5 * SpecialFunction::regularizedBeta($k / $scale, $k / 2.0, 0.5, $t * $t / $scale);
    }

    protected function guessUpperQuantile(float $p): float
    {
        $k = $this->degreesOfFreedom;
        $normal = new Normal()->upperQuantile($p);

        // Cornish-Fisher, the standard correction to the normal quantile. It
        // is good for many degrees of freedom and hopeless for few, which is
        // what the power law below is for.
        $corrected = $normal + ($normal ** 3 + $normal) / (4.0 * $k);

        if ($k > 8.0) {
            return $corrected;
        }

        // For few degrees of freedom the tail is a power law: P(T > t) tends
        // to c t^-k, so inverting that asymptote lands within a factor rather
        // than within eleven orders of magnitude.
        $logC = SpecialFunction::logGamma(($k + 1.0) / 2.0) - SpecialFunction::logGamma($k / 2.0)
            - 0.5 * log($k * M_PI) + log($k / 2.0) + ($k / 2.0 - 1.0) * log($k) - log($k / 2.0);

        return max($corrected, exp(($logC - log($p)) / $k));
    }

    /**
     * Required by the base class, and unreachable -- unlike the guess above,
     * which the generic inverse does use.
     *
     * The base returns the infimum for p = 1, and upperQuantile() sends every
     * p above a half to the reflection of its complement, so the base is only
     * ever asked for p below a half. p = 1 arrives here as -upperQuantile(0),
     * which is -INF by the other end of the same branch. The value below is
     * what that path produces anyway.
     */
    protected function infimum(): float
    {
        return -INF;
    }
}
