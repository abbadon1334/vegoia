<?php

declare(strict_types=1);

namespace Vegoia\Stats\Regression;

use function count;

use Vegoia\Exception\InvalidArgument;

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
}
