<?php

declare(strict_types=1);

namespace Vegoia\Tests\Reference\Stats;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Vegoia\Stats\OneWayAnova;
use Vegoia\Stats\Regression\Fit;
use Vegoia\Stats\Regression\LeastSquares;
use Vegoia\Tests\Support\Lre;
use Vegoia\Tests\Support\NistAnova;
use Vegoia\Tests\Support\NistRegression;
use Vegoia\Tests\Support\Paths;

/**
 * p-values, confidence intervals and prediction intervals.
 *
 * Split deliberately into two questions that are usually conflated.
 *
 * The first is whether the inference arithmetic is right. That is asked by
 * building a Fit from NIST's certified coefficients and standard errors and
 * checking what comes out, so the answer does not depend on how well anything
 * here fits a regression -- the reference is then as certified as NIST's own
 * numbers, on every dataset including Filip, whose design matrix has
 * numerical rank 10 out of 11 and which statsmodels can only fit through a
 * pseudo-inverse.
 *
 * The second is whether the whole path agrees with statsmodels: fit the data
 * with this library, then compare the intervals it reports. That one is
 * end-to-end and is held to a looser bar, because it inherits whatever the
 * fit gave up -- and it is skipped on Filip, where the reference itself is
 * an artefact of a fallback rather than an answer.
 *
 * @see tools/generate_inference_fixtures.py
 */
#[CoversClass(Fit::class)]
#[CoversClass(OneWayAnova::class)]
#[Group('reference')]
final class InferenceTest extends TestCase
{
    /** @var array<string, mixed>|null */
    private static ?array $fixture = null;

    /** @return array<string, mixed> */
    private static function section(string $name): array
    {
        if (self::$fixture === null) {
            /** @var array<string, mixed> $decoded */
            $decoded = json_decode(
                (string) file_get_contents(Paths::fixture('stats/inference.json')),
                true,
                flags: JSON_THROW_ON_ERROR,
            );
            self::$fixture = $decoded;
        }

        /** @var array<string, mixed> $out */
        $out = self::$fixture[$name];

        return $out;
    }

    /** @return iterable<string, array{string, array<string, mixed>}> */
    public static function regressions(): iterable
    {
        /** @var array<string, array<string, mixed>> $entries */
        $entries = self::section('regression');

        foreach ($entries as $name => $entry) {
            yield $name => [$name, $entry];
        }
    }

    /** @return iterable<string, array{string, array<string, mixed>}> */
    public static function analysesOfVariance(): iterable
    {
        /** @var array<string, array<string, mixed>> $entries */
        $entries = self::section('anova');

        foreach ($entries as $name => $entry) {
            yield $name => [$name, $entry];
        }
    }

    /**
     * The inference arithmetic, on certified inputs.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('regressions')]
    public function test_the_t_statistics_and_p_values_follow_from_the_certified_fit(
        string $name,
        array $entry,
    ): void {
        /**
         * @var array{
         *     certified_coefficients: list<float>, certified_standard_errors: list<float>,
         *     degrees_of_freedom: int, observations: int, parameters: int, has_intercept: bool,
         *     t_statistics: list<float|null>, p_values: list<float|null>,
         *     confidence_intervals: array<string, list<array{float, float}>>
         * } $entry
         */
        $fit = self::fitFromCertified($entry);

        foreach ($entry['t_statistics'] as $index => $expected) {
            if ($expected === null) {
                // A certified standard error of zero: the fit is exact, so the
                // coefficient is infinitely many standard errors from zero and
                // the p-value is exactly zero. Pinned rather than skipped --
                // a NAN here would be a bug, and two of these datasets are
                // exact fits.
                self::assertSame(INF, abs($fit->tStatistic($index)), "{$name}: t of B{$index}");
                self::assertSame(0.0, $fit->pValue($index), "{$name}: p of B{$index}");

                continue;
            }

            Lre::assertDigits($fit->tStatistic($index), $expected, "{$name}: t of B{$index}");

            /** @var float $p */
            $p = $entry['p_values'][$index];

            // The p-value is a tail probability of the t just checked, so its
            // accuracy is that of the t amplified by the tail's steepness.
            // Eleven digits is comfortably inside what that allows and well
            // past what anyone reports.
            Lre::assertDigits($fit->pValue($index), $p, "{$name}: p of B{$index}", 11.0);
        }
    }

    /**
     * Confidence intervals for the coefficients, on certified inputs.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('regressions')]
    public function test_the_confidence_intervals_follow_from_the_certified_fit(
        string $name,
        array $entry,
    ): void {
        /**
         * @var array{
         *     certified_coefficients: list<float>, certified_standard_errors: list<float>,
         *     degrees_of_freedom: int, observations: int, parameters: int, has_intercept: bool,
         *     confidence_intervals: array<string, list<array{float, float}>>
         * } $entry
         */
        $fit = self::fitFromCertified($entry);

        foreach ($entry['confidence_intervals'] as $level => $intervals) {
            foreach ($intervals as $index => [$low, $high]) {
                [$ourLow, $ourHigh] = $fit->confidenceInterval($index, (float) $level);

                Lre::assertDigits($ourLow, $low, "{$name}: B{$index} lower bound at {$level}", 12.0);
                Lre::assertDigits($ourHigh, $high, "{$name}: B{$index} upper bound at {$level}", 12.0);
            }
        }
    }

    /**
     * A confidence interval must contain the estimate and widen with the level.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('regressions')]
    public function test_a_higher_confidence_level_gives_a_wider_interval(
        string $name,
        array $entry,
    ): void {
        /**
         * @var array{
         *     certified_coefficients: list<float>, certified_standard_errors: list<float>,
         *     degrees_of_freedom: int, observations: int, parameters: int, has_intercept: bool,
         *     confidence_intervals: array<string, list<array{float, float}>>
         * } $entry
         */
        $fit = self::fitFromCertified($entry);

        foreach (array_keys($entry['certified_coefficients']) as $index) {
            [$low90, $high90] = $fit->confidenceInterval($index, 0.90);
            [$low99, $high99] = $fit->confidenceInterval($index, 0.99);
            $estimate = $fit->coefficients[$index];

            self::assertLessThanOrEqual($estimate, $low90, "{$name}: B{$index} below its own estimate");
            self::assertGreaterThanOrEqual($estimate, $high90, "{$name}: B{$index} above its own estimate");
            self::assertLessThanOrEqual($low90, $low99, "{$name}: B{$index} 99% reaches lower");
            self::assertGreaterThanOrEqual($high90, $high99, "{$name}: B{$index} 99% reaches higher");
        }
    }

    /**
     * The whole path: fit the data here, then compare what comes out.
     *
     * Split in two, because an interval's two halves answer to different
     * references and comparing the endpoints would hide both.
     *
     * The fitted value has a certified answer -- it is the certified
     * coefficients against the design row -- so it is checked against that,
     * with statsmodels' distance from the same certified value as the bar.
     * Being closer than statsmodels has to pass, and on these datasets it
     * does: at the first row of Wampler5 the truth is exactly 1, statsmodels
     * returns 1.0000011, and this library returns 1.0000000014.
     *
     * The half-width has no certified answer, since it needs a coefficient
     * covariance NIST does not certify, so it is compared against
     * statsmodels' directly and at a plainer bar.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('regressions')]
    public function test_the_prediction_intervals_agree_with_statsmodels(string $name, array $entry): void
    {
        /**
         * @var array{
         *     rank_deficient: bool,
         *     residual_standard_deviation: float|null,
         *     statsmodels?: array{predictions: list<array{
         *         row: int, design: list<float>, certified_fitted: string,
         *         statsmodels_fitted: float, statsmodels_fitted_accuracy: float|null,
         *         statsmodels_mean_half_width: float, statsmodels_prediction_half_width: float
         *     }>}
         * } $entry
         */
        if ($entry['rank_deficient']) {
            self::markTestSkipped(
                "{$name}: the design matrix is rank deficient, so statsmodels fits it through a "
                . 'pseudo-inverse and its intervals describe that fallback rather than the model. '
                . 'The fixture records nothing from it for that reason -- the numbers move '
                . 'between machines on identical software.'
            );
        }

        $fit = self::fitFromData($name);

        // NIST certifies a residual standard deviation of exactly zero for
        // Wampler1 and Wampler2: they are polynomials that pass through their
        // own data. Every interval there is a point, and what both libraries
        // report is their own residual noise rather than a width.
        $exactFit = $entry['residual_standard_deviation'] === 0.0;

        // Present on every dataset that reaches here: the rank-deficient ones
        // return above, and they are the only ones the fixture omits it for.
        if (! isset($entry['statsmodels'])) {
            self::fail("{$name}: the fixture omits statsmodels for a design that is not rank deficient");
        }

        foreach ($entry['statsmodels']['predictions'] as $prediction) {
            $design = $prediction['design'];
            $predictors = $fit->hasIntercept ? array_slice($design, 1) : $design;
            $row = $prediction['row'];

            $required = $prediction['statsmodels_fitted_accuracy'] === null
                ? (float) Lre::DEFAULT_DIGITS
                : min((float) Lre::DEFAULT_DIGITS, $prediction['statsmodels_fitted_accuracy'] - 0.5);

            Lre::assertDigits(
                $fit->predict($predictors),
                (float) $prediction['certified_fitted'],
                "{$name}: fitted value at row {$row}",
                $required,
            );

            // Half-widths, not endpoints: an endpoint is the fitted value plus
            // the half-width, so comparing endpoints would let the fitted
            // value's error stand in for the half-width's and the other way
            // round.
            [$low, $high] = $fit->meanResponseInterval($predictors);
            self::assertHalfWidth(
                ($high - $low) / 2.0,
                $prediction['statsmodels_mean_half_width'],
                $exactFit,
                "{$name}: half-width of the mean interval at row {$row}",
            );

            [$low, $high] = $fit->predictionInterval($predictors);
            self::assertHalfWidth(
                ($high - $low) / 2.0,
                $prediction['statsmodels_prediction_half_width'],
                $exactFit,
                "{$name}: half-width of the prediction interval at row {$row}",
            );
        }
    }

    /**
     * A prediction interval is wider than the interval for the mean it is
     * centred on, always, because a single observation scatters around the
     * line as well as the line being uncertain.
     */
    /** @param array<string, mixed> $entry */
    #[DataProvider('regressions')]
    public function test_a_prediction_interval_is_wider_than_the_mean_interval(
        string $name,
        array $entry,
    ): void {
        $set = NistRegression::load($name);

        if ($set->isPolynomial()) {
            self::markTestSkipped("{$name}: predict() takes the raw predictor for a polynomial fit");
        }

        $fit = self::fitFromData($name);
        $predictors = $set->predictors[0];

        [$meanLow, $meanHigh] = $fit->meanResponseInterval($predictors);
        [$low, $high] = $fit->predictionInterval($predictors);

        self::assertLessThan($meanLow, $low, "{$name}: prediction reaches lower");
        self::assertGreaterThan($meanHigh, $high, "{$name}: prediction reaches higher");
    }

    /** @param array<string, mixed> $entry */
    #[DataProvider('analysesOfVariance')]
    public function test_the_anova_p_value_agrees_with_scipy(string $name, array $entry): void
    {
        /** @var array{f_statistic: float, p_value: float, between_degrees_of_freedom: int, within_degrees_of_freedom: int} $entry */
        $anova = OneWayAnova::of(NistAnova::load($name)->groups);

        self::assertSame($entry['between_degrees_of_freedom'], $anova->betweenDegreesOfFreedom, $name);
        self::assertSame($entry['within_degrees_of_freedom'], $anova->withinDegreesOfFreedom, $name);

        Lre::assertDigits($anova->pValue(), $entry['p_value'], "{$name}: p-value", 11.0);
    }

    /**
     * The interval half-width, against statsmodels rather than against a
     * certified value, because NIST certifies no coefficient covariance and
     * so there is nothing better to compare with.
     *
     * Eight digits, and the number is measured rather than chosen: across the
     * nine datasets whose design matrix has full rank, agreement runs from
     * 13.8 digits on Norris down to 8.29 on Longley -- whose design is the
     * most ill-conditioned in the collection and the reason it was published.
     * The polynomials sit between, at ten to eleven. A floor at eight is what
     * the worst case actually supports; everything else clears it by five.
     *
     * The exact fits are a different question. NIST certifies their residual
     * standard deviation as zero, so the true half-width is zero and there is
     * nothing to agree with to any number of digits. What is asked there is
     * that this library's residual noise is no larger than statsmodels' --
     * and it is smaller, by a factor of 32 on Wampler1 and 8000 on Wampler2.
     */
    private static function assertHalfWidth(
        float $computed,
        float $reference,
        bool $exactFit,
        string $what,
    ): void {
        if ($exactFit) {
            self::assertLessThanOrEqual(
                abs($reference),
                abs($computed),
                "{$what}: the fit is exact, so the interval is a point and this is noise -- "
                . 'it may be smaller than the reference but not larger',
            );

            return;
        }

        Lre::assertDigits($computed, $reference, $what, 8.0);
    }

    /**
     * @param array{certified_coefficients: list<float>, certified_standard_errors: list<float>,
     *     degrees_of_freedom: int, observations: int, parameters: int, has_intercept: bool} $entry
     */
    private static function fitFromCertified(array $entry): Fit
    {
        return new Fit(
            $entry['certified_coefficients'],
            $entry['certified_standard_errors'],
            0.0,
            0.0,
            0.0,
            $entry['observations'],
            $entry['parameters'],
            $entry['degrees_of_freedom'],
            $entry['has_intercept'],
        );
    }

    private static function fitFromData(string $name): Fit
    {
        $set = NistRegression::load($name);

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
