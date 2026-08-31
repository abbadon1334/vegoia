<?php

declare(strict_types=1);

namespace Vegoia\Stats\Regression;

use function abs;
use function array_key_exists;
use function count;
use function is_nan;
use function max;
use function sqrt;

use Vegoia\Exception\InvalidArgument;
use Vegoia\Stats\Distribution\FisherSnedecor;
use Vegoia\Stats\Distribution\StudentT;
use Vegoia\Support\CompensatedSum;

/**
 * The result of a least squares fit.
 *
 * Standard errors are included rather than left to the caller because getting
 * them right requires the decomposition the fit already computed; recovering
 * them afterwards from the coefficients alone means re-deriving X'X, which is
 * the numerically ruinous step the fit went out of its way to avoid.
 */
final readonly class Fit
{
    /**
     * @param list<float> $coefficients   B0 first when the model has an intercept
     * @param list<float> $standardErrors parallel to $coefficients
     */
    public function __construct(
        public array $coefficients,
        public array $standardErrors,
        public float $residualStandardDeviation,
        public float $rSquared,
        public float $residualSumOfSquares,
        public int $observations,
        public int $parameters,
        public int $degreesOfFreedom,
        public bool $hasIntercept,
        /**
         * The full coefficient covariance, not only its diagonal.
         *
         * Needed whenever a linear combination of coefficients matters -- a
         * prediction interval, or a change of variable such as the polynomial
         * fit performs -- because those depend on the off-diagonal terms too.
         * Rescaling the standard errors alone would understate them wherever
         * the coefficients are correlated.
         *
         * @var list<list<float>>
         */
        public array $covariance = [],
        /**
         * The total sum of squares, about the mean when the model has an
         * intercept and about zero when it does not.
         *
         * Kept because the overall F test needs it. Forming F from R-squared
         * alone divides by 1 - R-squared, and a good fit makes that a
         * subtraction of nearly equal numbers: on the Norris dataset
         * R-squared is 0.99999999982 and nine digits go with it.
         */
        public float $totalSumOfSquares = 0.0,
    ) {
    }

    /**
     * Predict from one row of predictors, in the same layout the fit was given
     * (without the intercept column, which is added here when present).
     *
     * @param list<float> $predictors
     */
    public function predict(array $predictors): float
    {
        $expected = $this->parameters - ($this->hasIntercept ? 1 : 0);

        if (count($predictors) !== $expected) {
            throw InvalidArgument::malformedEdge(
                "predict() expects {$expected} predictors, " . count($predictors) . ' given'
            );
        }

        $offset = 0;
        $value = 0.0;

        if ($this->hasIntercept) {
            $value = $this->coefficients[0];
            $offset = 1;
        }

        foreach ($predictors as $index => $predictor) {
            $value += $this->coefficients[$index + $offset] * $predictor;
        }

        return $value;
    }

    /**
     * Student's t for one coefficient: the estimate over its standard error.
     *
     * Infinite when the standard error is zero, which is not a degenerate
     * case to be guarded against but the correct answer to an exact fit. Two
     * of the NIST datasets are exact polynomials with no residual at all, and
     * a coefficient known without error is infinitely many standard errors
     * from zero.
     */
    public function tStatistic(int $index): float
    {
        $error = $this->standardErrors[$this->assertCoefficient($index)];

        if ($error === 0.0) {
            return $this->coefficients[$index] === 0.0 ? NAN : INF * ($this->coefficients[$index] <=> 0);
        }

        return $this->coefficients[$index] / $error;
    }

    /**
     * The two-sided p-value for one coefficient against a null of zero.
     *
     * Two-sided because the question a regression asks is whether the
     * coefficient is zero, not whether it is below zero; doubling the upper
     * tail of |t| is what every statistical package reports for this and what
     * the reference fixture pins.
     */
    public function pValue(int $index): float
    {
        $t = $this->tStatistic($index);

        if (is_nan($t)) {
            return NAN;
        }

        return 2.0 * new StudentT((float) $this->degreesOfFreedom)->survival(abs($t));
    }

    /**
     * A two-sided confidence interval for one coefficient.
     *
     * @return array{float, float} the lower and upper bounds
     */
    public function confidenceInterval(int $index, float $level = 0.95): array
    {
        $margin = $this->criticalValue($level) * $this->standardErrors[$this->assertCoefficient($index)];
        $estimate = $this->coefficients[$index];

        return [$estimate - $margin, $estimate + $margin];
    }

    /**
     * The overall F: every slope at once, against a model with only an intercept.
     *
     * Undefined without an intercept -- there is no smaller model to compare
     * with -- and undefined with no slopes, for the same reason.
     */
    public function fStatistic(): float
    {
        $slopes = $this->assertOverallTestIsDefined();

        // From the sums of squares rather than from R-squared. Both are the
        // same formula rearranged, but the R-squared route divides by
        // 1 - R-squared, which cancels away precisely when the fit is good.
        $explained = ($this->totalSumOfSquares - $this->residualSumOfSquares) / $slopes;
        $unexplained = $this->residualSumOfSquares / $this->degreesOfFreedom;

        return $explained / $unexplained;
    }

    /** How often a fit this good arises when no predictor matters at all. */
    public function overallPValue(): float
    {
        $slopes = $this->assertOverallTestIsDefined();

        return new FisherSnedecor((float) $slopes, (float) $this->degreesOfFreedom)
            ->survival($this->fStatistic());
    }

    /**
     * A confidence interval for the mean response at one design point.
     *
     * Where the coefficient intervals ask about one parameter, this asks about
     * the fitted surface: given these predictors, where does the average
     * response lie? Its width comes from the whole covariance matrix through
     * x' C x, not from the diagonal, because a prediction is a linear
     * combination of correlated estimates -- adding the individual variances
     * would misstate it in both directions depending on the sign of the
     * correlations.
     *
     * @param list<float> $predictors
     *
     * @return array{float, float}
     */
    public function meanResponseInterval(array $predictors, float $level = 0.95): array
    {
        $fitted = $this->predict($predictors);
        $margin = $this->criticalValue($level) * sqrt($this->predictionVariance($predictors));

        return [$fitted - $margin, $fitted + $margin];
    }

    /**
     * A prediction interval for one new observation at the same design point.
     *
     * Wider than the mean interval, and by more than a little: the mean has
     * only the uncertainty of where the line is, while a single observation
     * also has to scatter around it. The extra term is the residual variance
     * itself, and it does not shrink as the sample grows -- with enough data
     * the mean interval collapses to a point and this one does not.
     *
     * @param list<float> $predictors
     *
     * @return array{float, float}
     */
    public function predictionInterval(array $predictors, float $level = 0.95): array
    {
        $fitted = $this->predict($predictors);
        $variance = $this->residualStandardDeviation ** 2 + $this->predictionVariance($predictors);
        $margin = $this->criticalValue($level) * sqrt($variance);

        return [$fitted - $margin, $fitted + $margin];
    }

    /**
     * x' C x: the variance of the fitted mean at one design point.
     *
     * @param list<float> $predictors
     */
    private function predictionVariance(array $predictors): float
    {
        if ($this->covariance === []) {
            throw InvalidArgument::malformedEdge(
                'an interval for a prediction needs the coefficient covariance, and this fit '
                . 'was constructed without it'
            );
        }

        /** @var list<float> $row */
        $row = $this->hasIntercept ? [1.0, ...$predictors] : $predictors;
        $variance = new CompensatedSum();

        foreach ($row as $i => $left) {
            foreach ($row as $j => $right) {
                $variance->add($left * $this->covariance[$i][$j] * $right);
            }
        }

        return max(0.0, $variance->value());
    }

    /** The two-sided t multiplier for a confidence level. */
    private function criticalValue(float $level): float
    {
        if (! ($level > 0.0 && $level < 1.0)) {
            throw InvalidArgument::outOfRange('a confidence level', $level, 0.0, 1.0);
        }

        return new StudentT((float) $this->degreesOfFreedom)->upperQuantile((1.0 - $level) / 2.0);
    }

    private function assertCoefficient(int $index): int
    {
        if (! array_key_exists($index, $this->coefficients)) {
            throw InvalidArgument::outOfRange('a coefficient index', $index, 0, $this->parameters - 1);
        }

        return $index;
    }

    /** @return int the number of slopes, when an overall test makes sense */
    private function assertOverallTestIsDefined(): int
    {
        $slopes = $this->parameters - 1;

        if (! $this->hasIntercept || $slopes < 1) {
            throw InvalidArgument::malformedEdge(
                'the overall F test compares the model with an intercept-only one, so it needs '
                . 'an intercept and at least one slope'
            );
        }

        return $slopes;
    }
}
