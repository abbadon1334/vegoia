<?php

declare(strict_types=1);

namespace Vegoia\Tests\Reference\Nist;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Stats\Regression\Fit;
use Vegoia\Stats\Regression\LeastSquares;
use Vegoia\Tests\Support\AttainableAccuracy;
use Vegoia\Tests\Support\Lre;
use Vegoia\Tests\Support\NistRegression;

/**
 * Linear least squares against the NIST Statistical Reference Datasets.
 *
 * These datasets are chosen to be hard. Longley is the classic economic
 * regression that broke the packages of its day; Wampler is a polynomial with
 * exact integer answers and a design matrix that grows monstrous; Filip is a
 * degree-10 polynomial with a condition number near 1e15, which NIST notes
 * many packages cannot fit at all.
 *
 * Anything solving via the normal equations fails here, and fails silently:
 * X'X squares the condition number, so Filip's becomes 1e30, far past what a
 * double can represent, and the fit comes back full of confident nonsense. The
 * implementation therefore uses Householder QR, and these tests are what
 * proves it.
 *
 * @see https://www.itl.nist.gov/div898/strd/lls/lls.shtml
 */
#[CoversClass(LeastSquares::class)]
#[Group('reference')]
#[Group('nist')]
final class LinearRegressionTest extends TestCase
{
    private const string CEILINGS = 'nist/attainable_lls.json';

    /** @return iterable<string, array{string}> */
    public static function datasets(): iterable
    {
        foreach (['Norris', 'NoInt1', 'NoInt2', 'Pontius', 'Longley',
                  'Wampler1', 'Wampler2', 'Wampler3', 'Wampler4', 'Wampler5',
                  'Filip'] as $name) {
            yield $name => [$name];
        }
    }

    #[DataProvider('datasets')]
    public function test_coefficients_match_the_certified_values(string $name): void
    {
        $set = NistRegression::load($name);
        $fit = self::fit($set);
        $required = AttainableAccuracy::required($name, 'coefficients', self::CEILINGS);

        foreach ($set->estimates as $index => $certified) {
            Lre::assertDigits(
                $fit->coefficients[$index],
                $certified,
                "{$name}: coefficient B{$index}",
                $required,
            );
        }
    }

    #[DataProvider('datasets')]
    public function test_standard_errors_match_the_certified_values(string $name): void
    {
        $set = NistRegression::load($name);
        $fit = self::fit($set);
        $required = AttainableAccuracy::required($name, 'standardErrors', self::CEILINGS);

        foreach ($set->errors as $index => $certified) {
            Lre::assertDigits(
                $fit->standardErrors[$index],
                $certified,
                "{$name}: standard error of B{$index}",
                $required,
            );
        }
    }

    #[DataProvider('datasets')]
    public function test_residual_standard_deviation_matches(string $name): void
    {
        $set = NistRegression::load($name);

        Lre::assertDigits(
            self::fit($set)->residualStandardDeviation,
            $set->residualStandardDeviation,
            "{$name}: residual standard deviation",
            AttainableAccuracy::required($name, 'residualStandardDeviation', self::CEILINGS),
        );
    }

    #[DataProvider('datasets')]
    public function test_r_squared_matches(string $name): void
    {
        $set = NistRegression::load($name);

        Lre::assertDigits(
            self::fit($set)->rSquared,
            $set->rSquared,
            "{$name}: R-squared",
            digits: 7,
        );
    }

    /**
     * Wampler1 is y = 1 + x + x^2 + ... + x^5 with integer data: the fit is
     * exact, every coefficient is 1, and the residuals are zero. A perfect fit
     * makes the residual standard deviation zero, which is the division a naive
     * implementation performs without checking.
     */
    public function test_a_perfect_fit_does_not_divide_by_zero(): void
    {
        $set = NistRegression::load('Wampler1');
        $fit = self::fit($set);

        self::assertSame(1.0, $set->rSquared, 'the fixture claims an exact fit');
        self::assertEqualsWithDelta(1.0, $fit->rSquared, 1.0e-9);
        self::assertEqualsWithDelta(0.0, $fit->residualStandardDeviation, 1.0e-6);
        self::assertTrue(is_finite($fit->rSquared));
    }

    /**
     * The domain mapping must be invisible from outside.
     *
     * A polynomial fit maps x onto [-1, 1] before raising it to powers, then
     * maps the coefficients back. That is a conditioning device, not a change
     * of model: the coefficients returned are those of the polynomial in x,
     * and predictions are made at the original x. If the mapping ever leaked
     * -- a coefficient left in t, an unscaled prediction -- this catches it
     * without reference to any certified value.
     */
    public function test_the_domain_mapping_does_not_change_the_model(): void
    {
        // y = 2 - 3x + x^2, fitted far from the origin where the raw
        // Vandermonde would be badly conditioned.
        $x = [];
        $y = [];

        for ($i = 0; $i < 40; $i++) {
            $value = 1000.0 + $i * 0.5;
            $x[] = $value;
            $y[] = 2.0 - 3.0 * $value + $value ** 2;
        }

        $fit = LeastSquares::polynomial($x, $y, degree: 2);

        self::assertEqualsWithDelta(2.0, $fit->coefficients[0], 1.0e-3, 'intercept in x');
        self::assertEqualsWithDelta(-3.0, $fit->coefficients[1], 1.0e-6, 'linear term in x');
        self::assertEqualsWithDelta(1.0, $fit->coefficients[2], 1.0e-9, 'quadratic term in x');

        // Predictions are made at the original scale.
        self::assertEqualsWithDelta(
            2.0 - 3.0 * 1005.0 + 1005.0 ** 2,
            $fit->predict([1005.0, 1005.0 ** 2]),
            1.0e-3,
        );
    }

    public function test_a_polynomial_needs_predictors_that_vary(): void
    {
        $this->expectException(InvalidArgument::class);

        LeastSquares::polynomial([3.0, 3.0, 3.0, 3.0], [1.0, 2.0, 3.0, 4.0], degree: 2);
    }

    /**
     * The covariance is the whole matrix, and it is what makes the
     * back-transformed standard errors right: a change of variable mixes
     * coefficients, so their uncertainties mix too.
     */
    public function test_the_fit_carries_a_full_covariance_matrix(): void
    {
        $fit = self::fit(NistRegression::load('Longley'));

        self::assertCount(7, $fit->covariance);

        foreach ($fit->covariance as $row) {
            self::assertCount(7, $row);
        }

        for ($i = 0; $i < 7; $i++) {
            self::assertEqualsWithDelta(
                $fit->standardErrors[$i] ** 2,
                $fit->covariance[$i][$i],
                abs($fit->covariance[$i][$i]) * 1.0e-9,
                "variance of B{$i} must be its standard error squared",
            );

            for ($j = 0; $j < 7; $j++) {
                self::assertEqualsWithDelta(
                    $fit->covariance[$i][$j],
                    $fit->covariance[$j][$i],
                    abs($fit->covariance[$i][$j]) * 1.0e-9,
                    'covariance must be symmetric',
                );
            }
        }
    }

    public function test_it_reports_the_shape_of_the_problem_it_solved(): void
    {
        $fit = self::fit(NistRegression::load('Longley'));

        self::assertSame(16, $fit->observations);
        self::assertSame(7, $fit->parameters);
        self::assertSame(9, $fit->degreesOfFreedom);
        self::assertCount(7, $fit->coefficients);
    }

    public function test_predictions_reproduce_the_response_on_an_exact_fit(): void
    {
        $set = NistRegression::load('Wampler1');
        $fit = self::fit($set);

        foreach ($set->response as $row => $expected) {
            $x = $set->predictors[$row][0];
            $powers = [];
            for ($k = 1; $k <= $set->degree(); $k++) {
                $powers[] = $x ** $k;
            }

            self::assertEqualsWithDelta($expected, $fit->predict($powers), 1.0e-3, "row {$row}");
        }
    }

    private static function fit(NistRegression $set): Fit
    {
        if ($set->isPolynomial()) {
            $x = [];
            foreach ($set->predictors as $row) {
                $x[] = $row[0];
            }

            return LeastSquares::polynomial($x, $set->response, $set->degree(), $set->hasIntercept);
        }

        return LeastSquares::fit($set->predictors, $set->response, $set->hasIntercept);
    }

    /**
     * A design whose columns are linearly dependent has no unique answer, and
     * saying so is the only honest thing to return.
     *
     * Two identical predictors is not a contrived case -- it is what happens
     * the moment somebody regresses two features that turn out to measure the
     * same thing. Before this guard worked, that returned coefficients of
     * -9.5e12 and +9.5e12 that cancel to something plausible, with an
     * R-squared of 0.9959 and no signal at all. The standard errors were
     * 2.9e14, which is the only place the trouble showed, and nothing looks at
     * those unless it already suspects something.
     *
     * The guards existed. They compared floats with ===, against a
     * decomposition that leaves a residue near 1e-16 rather than a clean zero,
     * so they only fired on the trivially exact cases -- a column of zeros --
     * where the arithmetic happens to come out right by accident.
     */
    public function test_a_rank_deficient_design_is_refused(): void
    {
        $this->expectException(InvalidArgument::class);
        $this->expectExceptionMessageMatches('/rank deficient|linearly dependent/i');

        LeastSquares::fit(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]],
            [1.0, 2.0, 3.1, 3.9, 5.2],
        );
    }

    /** A column that is a multiple of another is just as dependent. */
    public function test_a_proportional_column_is_refused(): void
    {
        $this->expectException(InvalidArgument::class);

        LeastSquares::fit(
            [[1.0, 3.0], [2.0, 6.0], [3.0, 9.0], [4.0, 12.0], [5.0, 15.0]],
            [1.0, 2.0, 3.1, 3.9, 5.2],
        );
    }

    /** And so is a column that duplicates the intercept. */
    public function test_a_constant_column_beside_an_intercept_is_refused(): void
    {
        $this->expectException(InvalidArgument::class);

        LeastSquares::fit(
            [[1.0, 7.0], [2.0, 7.0], [3.0, 7.0], [4.0, 7.0], [5.0, 7.0]],
            [1.0, 2.0, 3.1, 3.9, 5.2],
            true,
        );
    }

    /**
     * The threshold has to refuse the singular without refusing the merely
     * difficult, and the gap it sits in was measured rather than assumed.
     *
     * The smallest diagonal of R, relative to the largest, is 1.6e-16 on the
     * collinear design above -- machine epsilon, which is what a dependent
     * column leaves behind. On the hardest legitimate case in the NIST
     * collection it is 1.3e-5, and on Filip's degree-ten Vandermonde 2.4e-2.
     * Eleven orders of magnitude separate the two, and the tolerance sits in
     * the middle, so this is not a judgement call that could go either way.
     */
    public function test_the_difficult_datasets_are_still_accepted(): void
    {
        foreach (['Longley', 'Norris', 'Pontius', 'Wampler5', 'Filip'] as $name) {
            $fit = self::fit(NistRegression::load($name));

            self::assertNotEmpty($fit->coefficients, "{$name} was refused");
        }
    }
}
