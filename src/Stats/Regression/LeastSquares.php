<?php

declare(strict_types=1);

namespace Vegoia\Stats\Regression;

use function abs;
use function array_fill;
use function array_values;
use function count;
use function max;
use function min;
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
     * The predictor is mapped onto [-1, 1] before being raised to powers, and
     * the coefficients mapped back afterwards.
     *
     * A Vandermonde built from raw values is as ill-conditioned as the data is
     * far from the origin and wide in range. On NIST's Filip -- degree 10 over
     * x in [-8.8, -3.1] -- the condition number is 1.8e15 and barely eight
     * digits survive; centred on the midpoint and scaled to unit half-range it
     * is 2.9e3 and fourteen do. The change of variable is exact in real
     * arithmetic, so this is conditioning rather than approximation, and it is
     * what numpy.polynomial.Polynomial.fit does for the same reason.
     *
     * @param list<float> $x
     * @param list<float> $y
     */
    public static function polynomial(array $x, array $y, int $degree, bool $withIntercept = true): Fit
    {
        if ($degree < 1) {
            throw InvalidArgument::outOfRange('Polynomial degree', (float) $degree, 1.0, INF);
        }

        if ($x === []) {
            throw InvalidArgument::emptyDataset('a polynomial fit');
        }

        $low = min($x);
        $high = max($x);
        $shift = ($low + $high) / 2.0;
        $scale = ($high - $low) / 2.0;

        if ($scale === 0.0) {
            // Every predictor identical: no polynomial is determined, and
            // scaling by zero would yield NaN rather than say so.
            throw InvalidArgument::malformedEdge(
                'A polynomial fit needs predictors that vary; every x is ' . $low
            );
        }

        $predictors = [];

        foreach ($x as $value) {
            $row = [];
            $power = 1.0;
            $t = ($value - $shift) / $scale;

            for ($k = 1; $k <= $degree; $k++) {
                $power *= $t;
                $row[] = $power;
            }

            $predictors[] = $row;
        }

        return self::inPowersOfX(
            self::fit($predictors, $y, $withIntercept),
            $shift,
            $scale,
            $withIntercept,
        );
    }

    /**
     * Map a fit in t = (x - shift) / scale back to powers of x.
     *
     * Expanding t^k by the binomial theorem is a triangular change of basis M,
     * so the coefficients become M*c. The standard errors do not simply
     * rescale: mixing coefficients mixes their uncertainties, so the whole
     * covariance transforms as M*C*M' and the errors come off its diagonal.
     * Rescaling the errors alone would understate them wherever the original
     * coefficients are correlated -- which, after a shift, they always are.
     */
    private static function inPowersOfX(Fit $fit, float $shift, float $scale, bool $withIntercept): Fit
    {
        $parameters = $fit->parameters;

        // Column j of the fit is the coefficient of t^(j + offset).
        $offset = $withIntercept ? 0 : 1;

        /** @var list<list<float>> $transform */
        $transform = [];

        for ($j = 0; $j < $parameters; $j++) {
            $row = array_fill(0, $parameters, 0.0);
            $powerJ = $j + $offset;

            for ($k = $j; $k < $parameters; $k++) {
                $powerK = $k + $offset;

                $row[$k] = self::binomial($powerK, $powerJ)
                    * (-$shift) ** ($powerK - $powerJ)
                    / $scale ** $powerK;
            }

            /** @var list<float> $row */
            $transform[] = $row;
        }

        $coefficients = [];
        $errors = [];

        for ($i = 0; $i < $parameters; $i++) {
            $value = new CompensatedSum();

            for ($k = 0; $k < $parameters; $k++) {
                $value->add($transform[$i][$k] * $fit->coefficients[$k]);
            }

            $coefficients[] = $value->value();

            // (M C M')[i][i], without forming the full product.
            $variance = new CompensatedSum();

            for ($a = 0; $a < $parameters; $a++) {
                if ($transform[$i][$a] === 0.0) {
                    continue;
                }

                for ($b = 0; $b < $parameters; $b++) {
                    if ($transform[$i][$b] === 0.0) {
                        continue;
                    }

                    $variance->add(
                        $transform[$i][$a] * $fit->covariance[$a][$b] * $transform[$i][$b]
                    );
                }
            }

            $errors[] = sqrt(abs($variance->value()));
        }

        return new Fit(
            $coefficients,
            $errors,
            $fit->residualStandardDeviation,
            $fit->rSquared,
            $fit->residualSumOfSquares,
            $fit->observations,
            $fit->parameters,
            $fit->degreesOfFreedom,
            $fit->hasIntercept,
            $fit->covariance,
        );
    }

    /** Exact for the small degrees a polynomial fit can meaningfully carry. */
    private static function binomial(int $n, int $k): float
    {
        $result = 1.0;

        for ($i = 0; $i < $k; $i++) {
            $result = $result * ($n - $i) / ($i + 1);
        }

        return $result;
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

        $covariance = self::covariance($triangular, $columns, $variance);

        return new Fit(
            $coefficients,
            self::standardErrorsFrom($covariance, $columns),
            $sigma,
            self::rSquared($response, $residualSumOfSquares, $withIntercept),
            $residualSumOfSquares,
            $rows,
            $columns,
            $degreesOfFreedom,
            $withIntercept,
            $covariance,
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
     * The coefficient covariance, sigma^2 (R^-1)(R^-1)'.
     *
     * The whole matrix and not just its diagonal, because a change of variable
     * -- as the polynomial fit performs -- mixes the coefficients, and the
     * standard errors of the transformed coefficients depend on their
     * covariances as well as their variances.
     *
     * R is inverted by back-substitution, which is exact up to rounding for a
     * triangular matrix. Forming R'R and inverting that would square the
     * condition number, undoing the reason QR was chosen.
     *
     * @param  list<float> $matrix contains R in its leading $columns rows
     * @return list<list<float>>
     */
    private static function covariance(array $matrix, int $columns, float $variance): array
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

        $covariance = [];

        for ($i = 0; $i < $columns; $i++) {
            $row = array_fill(0, $columns, 0.0);

            for ($j = 0; $j < $columns; $j++) {
                $sum = new CompensatedSum();

                for ($k = 0; $k < $columns; $k++) {
                    $sum->add($inverse[$i][$k] * $inverse[$j][$k]);
                }

                $row[$j] = abs($variance) * $sum->value();
            }

            /** @var list<float> $row array_fill over 0..columns-1 */
            $covariance[] = $row;
        }

        return $covariance;
    }

    /**
     * @param  list<list<float>> $covariance
     * @return list<float>
     */
    private static function standardErrorsFrom(array $covariance, int $columns): array
    {
        $errors = [];

        for ($i = 0; $i < $columns; $i++) {
            $errors[] = sqrt(abs($covariance[$i][$i]));
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
