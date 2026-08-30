<?php

declare(strict_types=1);

namespace Vegoia\Tests\Reference\Nist;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
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
}
