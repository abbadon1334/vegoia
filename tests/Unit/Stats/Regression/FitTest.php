<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit\Stats\Regression;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Stats\Regression\Fit;
use Vegoia\Stats\Regression\LeastSquares;

/**
 * What a Fit refuses.
 *
 * Five of the library's unreached guards lived here, and they were unreached
 * for a structural reason rather than an oversight: every other test builds a
 * Fit by running a regression, which by construction produces a well-formed
 * one. The guards exist for the cases a caller can create by hand or by asking
 * a question the fit cannot answer, and nothing was creating them.
 */
#[CoversClass(Fit::class)]
final class FitTest extends TestCase
{
    private static function straightLine(): Fit
    {
        return LeastSquares::fit([[1.0], [2.0], [3.0], [4.0], [5.0]], [2.1, 3.9, 6.2, 7.8, 10.1]);
    }

    /** @return iterable<string, array{callable(): mixed, string}> */
    public static function refused(): iterable
    {
        yield 'predicting from too few predictors' => [
            static fn () => self::straightLine()->predict([]),
            'expects 1 predictors',
        ];

        yield 'predicting from too many predictors' => [
            static fn () => self::straightLine()->predict([1.0, 2.0]),
            'expects 1 predictors',
        ];

        yield 'a coefficient index past the end' => [
            static fn () => self::straightLine()->tStatistic(5),
            'a coefficient index',
        ];

        yield 'a negative coefficient index' => [
            static fn () => self::straightLine()->confidenceInterval(-1),
            'a coefficient index',
        ];

        yield 'a confidence level of zero' => [
            static fn () => self::straightLine()->confidenceInterval(0, 0.0),
            'a confidence level',
        ];

        yield 'a confidence level of one' => [
            static fn () => self::straightLine()->confidenceInterval(0, 1.0),
            'a confidence level',
        ];

        // The overall F removes the slopes and compares. A model that has
        // none is already the smaller model, so there is nothing to remove
        // and nothing to ask -- unlike a model without an intercept, which
        // has slopes and is tested against y = 0. See
        // test_the_overall_test_works_without_an_intercept below.
        yield 'the overall F with nothing but an intercept' => [
            static fn () => LeastSquares::fit(
                [[], [], [], []],
                [2.1, 3.9, 6.2, 7.8],
                true,
            )->fStatistic(),
            'at least one slope to remove',
        ];

        yield 'the overall p-value with nothing but an intercept' => [
            static fn () => LeastSquares::fit(
                [[], [], [], []],
                [2.1, 3.9, 6.2, 7.8],
                true,
            )->overallPValue(),
            'at least one slope to remove',
        ];

        // A Fit assembled by hand -- as the inference reference test does from
        // NIST's certified numbers -- has no covariance, and an interval for a
        // prediction is a quadratic form in it.
        yield 'a prediction interval without a covariance' => [
            static fn () => new Fit([1.0, 2.0], [0.1, 0.2], 1.0, 0.9, 1.0, 10, 2, 8, true)
                ->predictionInterval([1.0]),
            'was constructed without it',
        ];

        yield 'a mean response interval without a covariance' => [
            static fn () => new Fit([1.0, 2.0], [0.1, 0.2], 1.0, 0.9, 1.0, 10, 2, 8, true)
                ->meanResponseInterval([1.0]),
            'was constructed without it',
        ];
    }

    /**
     * @param callable(): mixed $call
     */
    #[DataProvider('refused')]
    public function test_it_refuses_what_it_cannot_answer(callable $call, string $identifies): void
    {
        $this->expectException(InvalidArgument::class);
        $this->expectExceptionMessageMatches('/' . preg_quote($identifies, '/') . '/i');

        $call();
    }

    /**
     * A model fitted through the origin has an overall F after all.
     *
     * It used to be refused, on the reasoning that the F compares against an
     * intercept-only model and there is no intercept to keep. But the smaller
     * model is the one with the slopes removed, which without an intercept is
     * y = 0 -- a real model, and the comparison every reference makes. NIST
     * certifies 15750.25 for NoInt1 and 298.6666666666667 for NoInt2, and
     * statsmodels reports the same; refusing made this library the odd one
     * out. It was inconsistent internally too, since the total sum of squares
     * beside it is already measured about zero for such a model.
     *
     * The values here are statsmodels', on a fit small enough to check by
     * hand: the slope is 59.7/30, the residual sum of squares 0.097 against an
     * uncentred total of 118.9, and one regression degree of freedom rather
     * than none.
     */
    public function test_the_overall_test_works_without_an_intercept(): void
    {
        $fit = LeastSquares::fit([[1.0], [2.0], [3.0], [4.0]], [2.1, 3.9, 6.2, 7.8], false);

        self::assertEqualsWithDelta(1.99, $fit->coefficients[0], 1.0e-13);
        self::assertEqualsWithDelta(118.9, $fit->totalSumOfSquares, 1.0e-12);
        self::assertEqualsWithDelta(0.097, $fit->residualSumOfSquares, 1.0e-13);

        self::assertEqualsWithDelta(3674.3195876288523, $fit->fStatistic(), 1.0e-9);
        self::assertEqualsWithDelta(9.891906611799298e-06, $fit->overallPValue(), 1.0e-17);

        // The F is measured on one regression degree of freedom, not zero: an
        // off-by-one there scales the statistic and still looks plausible.
        self::assertSame(3, $fit->degreesOfFreedom);
        self::assertEqualsWithDelta(
            $fit->fStatistic(),
            ($fit->totalSumOfSquares - $fit->residualSumOfSquares) / ($fit->residualSumOfSquares / 3.0),
            1.0e-6,
        );
    }

    /** The nearest arguments on the other side of each boundary. */
    public function test_it_accepts_the_values_just_inside(): void
    {
        $fit = self::straightLine();

        // Finite, not merely float-typed: a guard that fired one value too
        // early would return NAN here rather than change the type.
        self::assertTrue(is_finite($fit->predict([1.5])));
        self::assertTrue(is_finite($fit->tStatistic(0)));
        self::assertTrue(is_finite($fit->tStatistic(1)));
        self::assertTrue(is_finite($fit->fStatistic()));

        // A confidence level of essentially zero collapses the interval onto
        // the estimate; essentially one blows it open. Both are accepted, and
        // both stay ordered.
        [$low, $high] = $fit->confidenceInterval(0, 1.0e-9);
        self::assertEqualsWithDelta($fit->coefficients[0], $low, 1.0e-6);
        self::assertEqualsWithDelta($fit->coefficients[0], $high, 1.0e-6);

        [$low, $high] = $fit->confidenceInterval(1, 0.999999999);
        self::assertLessThan($high, $low);

        [$low, $high] = $fit->predictionInterval([1.5]);
        self::assertLessThan($fit->predict([1.5]), $low);
        self::assertGreaterThan($fit->predict([1.5]), $high);
    }

    /**
     * A model with an intercept and no slopes has nothing to test either --
     * the smaller model it would be compared against is the same model.
     */
    public function test_the_overall_test_needs_at_least_one_slope(): void
    {
        $this->expectException(InvalidArgument::class);
        $this->expectExceptionMessageMatches('/at least one slope/');

        // An intercept and nothing else: predictors are empty rows.
        new Fit([1.0], [0.1], 1.0, 0.0, 1.0, 10, 1, 9, true, [[0.01]], 1.0)->fStatistic();
    }
}
