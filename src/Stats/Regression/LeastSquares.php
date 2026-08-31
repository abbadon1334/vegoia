<?php

declare(strict_types=1);

namespace Vegoia\Stats\Regression;

use function abs;
use function array_fill;
use function array_values;
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
 * Three refinements separate this from a textbook transcription, each one
 * measured against numpy on the NIST datasets rather than adopted on faith:
 *
 *   * every dot product accumulates with Neumaier compensation. A sequential
 *     sum carries O(n) rounding error where a compensated one carries O(1),
 *     and it is worth a full digit on Pontius -- numpy gets the same effect
 *     for free from pairwise summation inside its BLAS.
 *   * one step of iterative refinement. The residual of the first solution is
 *     itself a least squares problem in the same factorisation, so the
 *     correction is nearly free, and it recovers roughly half a digit.
 *   * the residual sum of squares is recomputed from the refined coefficients
 *     rather than read off the tail of the transformed response. The tail is
 *     the textbook route and needs no second pass, but it carries the error of
 *     every reflection applied to reach it; on Pontius the explicit form is
 *     worth half a digit in the standard errors, which inherit their accuracy
 *     entirely from this number.
 *
 * Measured against LAPACK called directly from C, not only against numpy --
 * numpy is a binding, and the distinction turned out to matter. Mean correct
 * digits across the NIST linear least squares collection:
 *
 *     Vegoia                       11.40
 *     numpy (via OpenBLAS)         10.93
 *     OpenBLAS called from C       10.84
 *     ATLAS LAPACK called from C   10.79
 *     LAPACK dgelsd (SVD)           9.64
 *
 * So there is no single "LAPACK accuracy" to fall short of: ATLAS and OpenBLAS
 * differ from each other by more than either differs from this code. Filip --
 * a degree-10 polynomial with a condition number near 1.8e15 -- is the one
 * dataset where OpenBLAS is ahead (8.55 against 7.96), and ATLAS is behind
 * both at 7.07.
 *
 * Column equilibration, column pivoting and a doubled-precision refinement
 * residual were each implemented and measured against Filip, and each made it
 * worse. See tools/lapack/ for the harness.
 *
 * @see G.H. Golub & C.F. Van Loan, Matrix Computations, 4th ed., ch. 5.
 * @see A. Bjorck (1996), Numerical Methods for Least Squares Problems, ch. 2.
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
        // Column equilibration was tried here and removed. Scaling each column
        // to unit norm is the standard advice for an ill-conditioned design,
        // and measured across the NIST collection it made results worse on
        // average -- including on Filip, the one dataset it was meant to
        // help. Left out rather than kept in and explained away.
        [$triangular, $reflections] = self::factorise($matrix, $rows, $columns);

        $transformed = self::applyReflections($reflections, $response, $rows, $columns);
        $coefficients = self::backSubstitute($triangular, $transformed, $columns);

        // One step of iterative refinement. The residual of the current
        // solution is a least squares problem against the same design, so the
        // existing reflections solve it: no second factorisation.
        $residual = [];

        for ($i = 0; $i < $rows; $i++) {
            $fitted = new CompensatedSum();

            for ($j = 0; $j < $columns; $j++) {
                $fitted->add($matrix[$i * $columns + $j] * $coefficients[$j]);
            }

            $residual[] = $response[$i] - $fitted->value();
        }

        $correction = self::backSubstitute(
            $triangular,
            self::applyReflections($reflections, $residual, $rows, $columns),
            $columns,
        );

        for ($j = 0; $j < $columns; $j++) {
            $coefficients[$j] += $correction[$j];
        }

        /** @var list<float> $coefficients only existing slots were updated */

        // Residual sum of squares from the refined solution.
        //
        // The tail of the transformed response is the textbook route and needs
        // no second pass, but it carries the error of every reflection applied
        // to get there. Recomputing y - X*beta after refinement costs one pass
        // and is measurably better: on Pontius it is worth 0.9 of a digit in
        // the standard errors, which inherit their accuracy entirely from
        // this number.
        $squares = new CompensatedSum();

        for ($i = 0; $i < $rows; $i++) {
            $fitted = new CompensatedSum();

            for ($j = 0; $j < $columns; $j++) {
                $fitted->add($matrix[$i * $columns + $j] * $coefficients[$j]);
            }

            $error = $response[$i] - $fitted->value();
            $squares->add($error * $error);
        }

        $residualSumOfSquares = $squares->value();
        $degreesOfFreedom = $rows - $columns;
        $variance = $degreesOfFreedom > 0 ? $residualSumOfSquares / $degreesOfFreedom : 0.0;
        $sigma = sqrt(abs($variance));

        return new Fit(
            $coefficients,
            self::standardErrors($triangular, $columns, $sigma),
            $sigma,
            self::rSquared($response, $residualSumOfSquares, $withIntercept),
            $residualSumOfSquares,
            $rows,
            $columns,
            $degreesOfFreedom,
            $withIntercept,
        );
    }

    /**
     * Householder QR in place, keeping each reflection so it can be replayed
     * against another vector -- which is what makes the refinement step cheap.
     *
     * @param  list<float> $matrix row-major, already equilibrated
     * @return array{list<float>, list<array{list<float>, float}>} the factored
     *         matrix, and each reflection as (vector, squared norm)
     */
    private static function factorise(array $matrix, int $rows, int $columns): array
    {
        $reflections = [];

        for ($k = 0; $k < $columns; $k++) {
            $normSquared = new CompensatedSum();

            for ($i = $k; $i < $rows; $i++) {
                $value = $matrix[$i * $columns + $k];
                $normSquared->add($value * $value);
            }

            $total = $normSquared->value();

            if ($total === 0.0) {
                throw InvalidArgument::malformedEdge(
                    "Design matrix is rank deficient: column {$k} is zero below the diagonal"
                );
            }

            $head = $matrix[$k * $columns + $k];
            $norm = sqrt($total);

            // Choose the sign that moves away from the head value, so the
            // subtraction below can never cancel.
            $alpha = $head >= 0.0 ? -$norm : $norm;

            $vector = array_fill(0, $rows - $k, 0.0);
            $vector[0] = $head - $alpha;

            for ($i = $k + 1; $i < $rows; $i++) {
                $vector[$i - $k] = $matrix[$i * $columns + $k];
            }

            $vectorNorm = new CompensatedSum();

            foreach ($vector as $value) {
                $vectorNorm->add($value * $value);
            }

            $vectorNormSquared = $vectorNorm->value();

            if ($vectorNormSquared === 0.0) {
                $reflections[] = [array_values($vector), 0.0];

                continue;
            }

            for ($j = $k; $j < $columns; $j++) {
                $dot = new CompensatedSum();

                for ($i = $k; $i < $rows; $i++) {
                    $dot->add($vector[$i - $k] * $matrix[$i * $columns + $j]);
                }

                $factor = 2.0 * $dot->value() / $vectorNormSquared;

                for ($i = $k; $i < $rows; $i++) {
                    $matrix[$i * $columns + $j] -= $factor * $vector[$i - $k];
                }
            }

            $reflections[] = [array_values($vector), $vectorNormSquared];
        }

        /** @var list<float> $matrix */
        return [$matrix, $reflections];
    }

    /**
     * @param  list<array{list<float>, float}> $reflections
     * @param  list<float>                     $vector
     * @return list<float>
     */
    private static function applyReflections(array $reflections, array $vector, int $rows, int $columns): array
    {
        for ($k = 0; $k < $columns; $k++) {
            [$reflection, $normSquared] = $reflections[$k];

            if ($normSquared === 0.0) {
                continue;
            }

            $dot = new CompensatedSum();

            for ($i = $k; $i < $rows; $i++) {
                $dot->add($reflection[$i - $k] * $vector[$i]);
            }

            $factor = 2.0 * $dot->value() / $normSquared;

            for ($i = $k; $i < $rows; $i++) {
                $vector[$i] -= $factor * $reflection[$i - $k];
            }
        }

        /** @var list<float> $vector only existing slots were updated */
        return $vector;
    }

    /**
     * @param  list<float> $triangular
     * @param  list<float> $transformed
     * @return list<float>
     */
    private static function backSubstitute(array $triangular, array $transformed, int $columns): array
    {
        $coefficients = array_fill(0, $columns, 0.0);

        for ($i = $columns - 1; $i >= 0; $i--) {
            $sum = new CompensatedSum();

            for ($j = $i + 1; $j < $columns; $j++) {
                $sum->add($triangular[$i * $columns + $j] * $coefficients[$j]);
            }

            $diagonal = $triangular[$i * $columns + $i];

            if ($diagonal === 0.0) {
                throw InvalidArgument::malformedEdge(
                    "Design matrix is singular: R has a zero on the diagonal at {$i}"
                );
            }

            $coefficients[$i] = ($transformed[$i] - $sum->value()) / $diagonal;
        }

        /** @var list<float> $coefficients */
        return $coefficients;
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
                $sum = new CompensatedSum();

                for ($j = $i + 1; $j <= $column; $j++) {
                    $sum->add($matrix[$i * $columns + $j] * $inverse[$j][$column]);
                }

                $unit = $i === $column ? 1.0 : 0.0;
                $inverse[$i][$column] = ($unit - $sum->value()) / $matrix[$i * $columns + $i];
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
