<?php

declare(strict_types=1);

namespace Vegoia\Stats;

use function abs;
use function count;
use function is_nan;
use function sqrt;

use Vegoia\Exception\InvalidArgument;
use Vegoia\Stats\Distribution\StudentT;

/**
 * Do two samples come from populations with the same mean?
 *
 * Two variants, and the choice between them is about a assumption rather than
 * about precision. Student's pools the two variances into one estimate, which
 * is right when the populations share a variance and wrong in a direction that
 * depends on which sample was larger when they do not. Welch's assumes
 * nothing, costs almost nothing when the variances are equal, and is what R's
 * `t.test` defaults to.
 *
 * The variances come from Descriptive, which is the reuse that matters most
 * here: the NIST NumAcc datasets exist to break exactly the naive one-pass
 * variance that a hand-rolled t-test would reach for, and Descriptive has
 * already paid for the corrected two-pass.
 */
final readonly class TTest
{
    private function __construct(
        public float $statistic,
        public float $degreesOfFreedom,
        public float $meanDifference,
        public float $standardError,
        public int $firstCount,
        public int $secondCount,
        public bool $pooled,
    ) {
    }

    /**
     * Student's two-sample t, pooling the two variances.
     *
     * @param list<float> $x
     * @param list<float> $y
     */
    public static function student(array $x, array $y): self
    {
        [$nx, $ny, $meanDifference, $varianceX, $varianceY] = self::describe($x, $y);

        $pooled = (($nx - 1) * $varianceX + ($ny - 1) * $varianceY) / ($nx + $ny - 2);
        $standardError = sqrt($pooled * (1.0 / $nx + 1.0 / $ny));

        return new self(
            self::statistic($meanDifference, $standardError),
            (float) ($nx + $ny - 2),
            $meanDifference,
            $standardError,
            $nx,
            $ny,
            true,
        );
    }

    /**
     * Welch's t, with the Satterthwaite degrees of freedom.
     *
     * @param list<float> $x
     * @param list<float> $y
     */
    public static function welch(array $x, array $y): self
    {
        [$nx, $ny, $meanDifference, $varianceX, $varianceY] = self::describe($x, $y);

        $termX = $varianceX / $nx;
        $termY = $varianceY / $ny;
        $standardError = sqrt($termX + $termY);

        // Welch-Satterthwaite. Zero over zero when both samples are constant,
        // which is the case where the statistic is not defined either.
        $denominator = $termX * $termX / ($nx - 1) + $termY * $termY / ($ny - 1);
        $degreesOfFreedom = $denominator > 0.0
            ? ($termX + $termY) ** 2 / $denominator
            : (float) ($nx + $ny - 2);

        return new self(
            self::statistic($meanDifference, $standardError),
            $degreesOfFreedom,
            $meanDifference,
            $standardError,
            $nx,
            $ny,
            false,
        );
    }

    /**
     * The two-sided p-value against a null of no difference in means.
     *
     * Two-sided because the question is whether the means differ, not whether
     * one is below the other; doubling the upper tail of |t| is what every
     * package reports for this, and what Fit::pValue() already does here.
     */
    public function pValue(): float
    {
        if (is_nan($this->statistic)) {
            return NAN;
        }

        if ($this->statistic === 0.0) {
            return 1.0;
        }

        return 2.0 * new StudentT($this->degreesOfFreedom)->survival(abs($this->statistic));
    }

    /**
     * A two-sided confidence interval for the difference in means.
     *
     * @return array{float, float}
     */
    public function confidenceInterval(float $level = 0.95): array
    {
        if (! ($level > 0.0 && $level < 1.0)) {
            throw InvalidArgument::outOfRange('A confidence level', $level, 0.0, 1.0);
        }

        $margin = new StudentT($this->degreesOfFreedom)->upperQuantile((1.0 - $level) / 2.0)
            * $this->standardError;

        return [$this->meanDifference - $margin, $this->meanDifference + $margin];
    }

    /**
     * @param list<float> $x
     * @param list<float> $y
     *
     * @return array{int, int, float, float, float}
     */
    private static function describe(array $x, array $y): array
    {
        $nx = count($x);
        $ny = count($y);

        if ($nx < 2) {
            throw InvalidArgument::tooFewValues('A two-sample t-test needs a first sample that', $nx, 2);
        }

        if ($ny < 2) {
            throw InvalidArgument::tooFewValues('A two-sample t-test needs a second sample that', $ny, 2);
        }

        $first = Descriptive::of($x);
        $second = Descriptive::of($y);

        // mean(x) - mean(y), first minus second, which fixes the sign of the
        // statistic. When both means are large and close this cancels, and
        // nothing can be done about that beyond making the means themselves
        // right -- the difference IS the signal.
        return [$nx, $ny, $first->mean() - $second->mean(), $first->variance(), $second->variance()];
    }

    /**
     * A zero standard error is an exactly determined difference, not an error:
     * infinitely many standard errors from zero, or undefined when the
     * difference is zero too. Fit::tStatistic() takes the same position.
     */
    private static function statistic(float $meanDifference, float $standardError): float
    {
        if ($standardError > 0.0) {
            return $meanDifference / $standardError;
        }

        return $meanDifference === 0.0 ? NAN : INF * ($meanDifference <=> 0.0);
    }
}
