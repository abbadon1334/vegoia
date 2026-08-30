<?php

declare(strict_types=1);

namespace Vegoia\Stats\Regression;

use function abs;
use function array_fill;
use function count;
use function sqrt;

use Vegoia\Exception\InvalidArgument;
use Vegoia\Stats\Descriptive;
use Vegoia\Support\CompensatedSum;

/**
 * Ordinary least squares by Householder QR.
 *
 * The obvious way to fit a linear model is to solve the normal equations,
 * X'X b = X'y. It is short, it is what the textbook derivation hands you, and
 * it should almost never be used: forming X'X squares the condition number of
 * the problem. A design matrix with condition 1e8 -- unremarkable for a cubic
 * fit -- becomes 1e16 in the normal equations, past what a double can hold, and
 * the solver returns a confident answer with no correct digits and no warning.
 *
 * Householder QR factors X = QR directly and never forms X'X, so the condition
 * number is not squared. Two things then come out almost for free:
 *
 *   * applying the same reflections to y leaves the residual sum of squares as
 *     the squared norm of its tail, so no separate residual pass is needed;
 *   * the covariance of the coefficients is sigma^2 (R^-1)(R^-1)', obtained by
 *     inverting a triangular matrix -- again without touching X'X. Computing
 *     the standard errors from R'R instead would reintroduce the very squaring
 *     the decomposition was chosen to avoid, and on NIST's Filip it produces
 *     negative variances.
 *
 * @see G.H. Golub & C.F. Van Loan, Matrix Computations, 4th ed., ch. 5.
 */
final class LeastSquares
{
    /**
     * @param list<list<float>> $predictors one row per observation, without an
     *                                      intercept column
     * @param list<float>       $response
     */
    public static function fit(array $predictors, array $response, bool $withIntercept = true): Fit
    {
        $observations = count($response);

        if ($observations === 0) {
            throw InvalidArgument::emptyDataset('a regression');
        }

        if (count($predictors) !== $observations) {
            throw InvalidArgument::malformedEdge(
                'Design matrix has ' . count($predictors) . " rows but the response has {$observations}"
            );
        }

        $columns = count($predictors[0]) + ($withIntercept ? 1 : 0);

        if ($columns === 0) {
            throw InvalidArgument::tooFewValues('A regression', 0, 1);
        }

        if ($observations < $columns) {
            throw InvalidArgument::tooFewValues(
                'A regression with ' . $columns . ' parameters',
                $observations,
                $columns,
            );
        }

        // Row-major dense matrix: the decomposition sweeps columns within a
        // row, so contiguous rows keep the access pattern linear.
        $matrix = [];

        foreach ($predictors as $row => $values) {
            if (count($values) !== $columns - ($withIntercept ? 1 : 0)) {
                throw InvalidArgument::malformedEdge("Design row {$row} has the wrong width");
            }

            if ($withIntercept) {
                $matrix[] = 1.0;
            }

            foreach ($values as $value) {
                $matrix[] = $value;
            }
        }

        return self::solve($matrix, $response, $observations, $columns, $withIntercept);
    }

    /**
     * Fit y = B0 + B1*x + ... + Bd*x^d.
     *
     * Kept separate because building the powers is where a polynomial fit
     * usually goes wrong, and because the resulting design matrix is a
     * Vandermonde -- notoriously ill-conditioned, and the reason this class
     * decomposes rather than solving normal equations.
     *
     * @param list<float> $x
     * @param list<float> $y
     */
    public static function polynomial(array $x, array $y, int $degree, bool $withIntercept = true): Fit
    {
        if ($degree < 1) {
            throw InvalidArgument::outOfRange('Polynomial degree', (float) $degree, 1.0, INF);
        }

        $predictors = [];

        foreach ($x as $value) {
            $row = [];
            $power = 1.0;

            for ($k = 1; $k <= $degree; $k++) {
                $power *= $value;
                $row[] = $power;
            }

            $predictors[] = $row;
        }

        return self::fit($predictors, $y, $withIntercept);
    }

    /**
     * @param list<float> $matrix row-major, $rows x $columns
     * @param list<float> $response
     */
    private static function solve(
        array $matrix,
        array $response,
        int $rows,
        int $columns,
        bool $withIntercept,
    ): Fit {
        $rhs = $response;

        // Householder reflections: each one zeroes everything below the
        // diagonal in one column, applied to the matrix and the response
        // together so the transformed response carries the residual with it.
        for ($k = 0; $k < $columns; $k++) {
            $normSquared = 0.0;

            for ($i = $k; $i < $rows; $i++) {
                $value = $matrix[$i * $columns + $k];
                $normSquared += $value * $value;
            }

            if ($normSquared === 0.0) {
                throw InvalidArgument::malformedEdge(
                    "Design matrix is rank deficient: column {$k} is zero below the diagonal"
                );
            }

            $head = $matrix[$k * $columns + $k];
            $norm = sqrt($normSquared);

            // Choose the sign that moves away from the head value, so the
            // subtraction below can never cancel.
            $alpha = $head >= 0.0 ? -$norm : $norm;

            $vector = array_fill(0, $rows - $k, 0.0);
            $vector[0] = $head - $alpha;

            for ($i = $k + 1; $i < $rows; $i++) {
                $vector[$i - $k] = $matrix[$i * $columns + $k];
            }

            $vectorNormSquared = 0.0;

            foreach ($vector as $value) {
                $vectorNormSquared += $value * $value;
            }

            if ($vectorNormSquared === 0.0) {
                continue;
            }

            for ($j = $k; $j < $columns; $j++) {
                $dot = 0.0;

                for ($i = $k; $i < $rows; $i++) {
                    $dot += $vector[$i - $k] * $matrix[$i * $columns + $j];
                }

                $scale = 2.0 * $dot / $vectorNormSquared;

                for ($i = $k; $i < $rows; $i++) {
                    $matrix[$i * $columns + $j] -= $scale * $vector[$i - $k];
                }
            }

            $dot = 0.0;

            for ($i = $k; $i < $rows; $i++) {
                $dot += $vector[$i - $k] * $rhs[$i];
            }

            $scale = 2.0 * $dot / $vectorNormSquared;

            for ($i = $k; $i < $rows; $i++) {
                $rhs[$i] -= $scale * $vector[$i - $k];
            }
        }

        // Back-substitution on the triangular R.
        $coefficients = array_fill(0, $columns, 0.0);

        for ($i = $columns - 1; $i >= 0; $i--) {
            $sum = $rhs[$i];

            for ($j = $i + 1; $j < $columns; $j++) {
                $sum -= $matrix[$i * $columns + $j] * $coefficients[$j];
            }

            $diagonal = $matrix[$i * $columns + $i];

            if ($diagonal === 0.0) {
                throw InvalidArgument::malformedEdge(
                    "Design matrix is singular: R has a zero on the diagonal at {$i}"
                );
            }

            $coefficients[$i] = $sum / $diagonal;
        }

        // The reflections leave the residual in the tail of the transformed
        // response, so its squared norm is the residual sum of squares -- no
        // second pass over the data, and no cancellation from subtracting
        // fitted values from observed ones.
        $residual = new CompensatedSum();

        for ($i = $columns; $i < $rows; $i++) {
            $residual->add($rhs[$i] * $rhs[$i]);
        }

        $residualSumOfSquares = $residual->value();
        $degreesOfFreedom = $rows - $columns;
        $variance = $degreesOfFreedom > 0 ? $residualSumOfSquares / $degreesOfFreedom : 0.0;

        /**
         * @var list<float> $matrix       reflections only overwrite existing slots
         * @var list<float> $coefficients back-substitution fills 0..columns-1
         */
        return new Fit(
            $coefficients,
            self::standardErrors($matrix, $columns, sqrt(abs($variance))),
            sqrt(abs($variance)),
            self::rSquared($response, $residualSumOfSquares, $withIntercept),
            $residualSumOfSquares,
            $rows,
            $columns,
            $degreesOfFreedom,
            $withIntercept,
        );
    }

    /**
     * se_i = sigma * ||row i of R^-1||.
     *
     * R is inverted column by column with back-substitution -- triangular, so
     * exact up to rounding -- rather than by forming and inverting R'R.
     *
     * @param  list<float> $matrix contains R in its leading $columns rows
     * @return list<float>
     */
    private static function standardErrors(array $matrix, int $columns, float $sigma): array
    {
        /** @var list<list<float>> $inverse */
        $inverse = [];

        for ($i = 0; $i < $columns; $i++) {
            $inverse[] = array_fill(0, $columns, 0.0);
        }

        for ($column = 0; $column < $columns; $column++) {
            for ($i = $column; $i >= 0; $i--) {
                $sum = $i === $column ? 1.0 : 0.0;

                for ($j = $i + 1; $j <= $column; $j++) {
                    $sum -= $matrix[$i * $columns + $j] * $inverse[$j][$column];
                }

                $inverse[$i][$column] = $sum / $matrix[$i * $columns + $i];
            }
        }

        $errors = [];

        for ($i = 0; $i < $columns; $i++) {
            $sum = new CompensatedSum();

            for ($j = 0; $j < $columns; $j++) {
                $sum->add($inverse[$i][$j] * $inverse[$i][$j]);
            }

            $errors[] = $sigma * sqrt($sum->value());
        }

        return $errors;
    }

    /**
     * Without an intercept the total sum of squares is measured about zero
     * rather than about the mean -- the convention NIST certifies against, and
     * the one that keeps R-squared in [0, 1] for such a model.
     *
     * @param list<float> $response
     */
    private static function rSquared(array $response, float $residualSumOfSquares, bool $withIntercept): float
    {
        $total = new CompensatedSum();

        if ($withIntercept) {
            $mean = Descriptive::of($response)->mean();

            foreach ($response as $value) {
                $total->add(($value - $mean) ** 2);
            }
        } else {
            foreach ($response as $value) {
                $total->add($value * $value);
            }
        }

        $totalSumOfSquares = $total->value();

        if ($totalSumOfSquares === 0.0) {
            return 1.0;
        }

        return 1.0 - $residualSumOfSquares / $totalSumOfSquares;
    }
}
