<?php

declare(strict_types=1);

namespace Vegoia\Tests\Reference\Stats;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Vegoia\Stats\Distribution\ChiSquared;
use Vegoia\Stats\Distribution\Distribution;
use Vegoia\Stats\Distribution\FisherSnedecor;
use Vegoia\Stats\Distribution\Normal;
use Vegoia\Stats\Distribution\StudentT;
use Vegoia\Tests\Support\Lre;
use Vegoia\Tests\Support\Paths;

/**
 * The four distributions, against a certified value and a measured ceiling.
 *
 * mpmath at 50 digits gives the truth, built from the integrals rather than
 * from any library's answer; SciPy is measured against it and its accuracy is
 * the ceiling; this implementation is required to come within half a digit of
 * SciPy, capped at 13.
 *
 * Both tails are checked separately at every point, and that is the substance
 * of this file rather than a detail. A p-value is a tail probability. At
 * z = 10 the survival function is 7.6e-24 and the cumulative is 1 to every
 * bit a double has, so an implementation that computes one by subtracting the
 * other from one returns zero and reports p = 0 for everything past six
 * sigma. The same for the quantiles: an inverse routed through the cumulative
 * cannot return the 1e-300 point.
 *
 * @see tools/generate_distribution_fixtures.py
 */
#[CoversClass(Normal::class)]
#[CoversClass(StudentT::class)]
#[CoversClass(ChiSquared::class)]
#[CoversClass(FisherSnedecor::class)]
#[Group('reference')]
final class DistributionTest extends TestCase
{
    /** How far below SciPy this implementation may land, in digits. */
    private const float MARGIN = 0.5;

    /** @var array<string, mixed>|null */
    private static ?array $fixture = null;

    /** @return array<string, mixed> */
    private static function section(string $name): array
    {
        if (self::$fixture === null) {
            /** @var array<string, mixed> $decoded */
            $decoded = json_decode(
                (string) file_get_contents(Paths::fixture('stats/distributions.json')),
                true,
                flags: JSON_THROW_ON_ERROR,
            );
            self::$fixture = $decoded;
        }

        /** @var array<string, mixed> $out */
        $out = self::$fixture[$name];

        return $out;
    }

    /** @param array{certified: string, scipy: float, attainable: float|null, vanishes?: bool} $entry */
    private static function assertAgreesWithScipy(float $computed, array $entry, string $what): void
    {
        if ($entry['vanishes'] ?? false) {
            self::assertSame(0.0, $computed, "{$what}: the true value is below any double");

            return;
        }

        $required = $entry['attainable'] === null
            ? (float) Lre::DEFAULT_DIGITS
            : min((float) Lre::DEFAULT_DIGITS, $entry['attainable'] - self::MARGIN);

        Lre::assertDigits($computed, (float) $entry['certified'], $what, $required);
    }

    /**
     * Every distribution in the fixture, flattened to (label, distribution,
     * points, quantiles) so the four share one pair of tests rather than four.
     *
     * @return iterable<string, array{Distribution, array<string, mixed>, array<string, mixed>}>
     */
    public static function distributions(): iterable
    {
        /** @var array{points: array<string, mixed>, upper_quantiles: array<string, mixed>} $normal */
        $normal = self::section('normal');
        yield 'normal' => [new Normal(), $normal['points'], $normal['upper_quantiles']];

        /** @var array<string, array{points: array<string, mixed>, upper_quantiles: array<string, mixed>}> $t */
        $t = self::section('student_t');
        foreach ($t as $df => $block) {
            yield "t({$df})" => [new StudentT((float) $df), $block['points'], $block['upper_quantiles']];
        }

        /** @var array<string, array{points: array<string, mixed>, upper_quantiles: array<string, mixed>}> $chi */
        $chi = self::section('chi_squared');
        foreach ($chi as $df => $block) {
            yield "chi2({$df})" => [new ChiSquared((float) $df), $block['points'], $block['upper_quantiles']];
        }

        /** @var array<string, array{points: array<string, mixed>, upper_quantiles: array<string, mixed>}> $f */
        $f = self::section('fisher');
        foreach ($f as $pair => $block) {
            [$d1, $d2] = explode(',', $pair);
            yield "F({$pair})" => [
                new FisherSnedecor((float) $d1, (float) $d2),
                $block['points'],
                $block['upper_quantiles'],
            ];
        }
    }

    /**
     * @param array<string, mixed> $points
     * @param array<string, mixed> $quantiles
     */
    #[DataProvider('distributions')]
    public function test_the_density_and_both_tails_match_the_certified_values(
        Distribution $distribution,
        array $points,
        array $quantiles,
    ): void {
        /** @var array<string, array<string, array{certified: string, scipy: float, attainable: float|null, vanishes?: bool}>> $points */
        foreach ($points as $x => $expected) {
            $value = (float) $x;

            self::assertAgreesWithScipy($distribution->density($value), $expected['pdf'], "pdf({$x})");
            self::assertAgreesWithScipy($distribution->cumulative($value), $expected['cdf'], "cdf({$x})");
            self::assertAgreesWithScipy($distribution->survival($value), $expected['sf'], "sf({$x})");
        }
    }

    /**
     * @param array<string, mixed> $points
     * @param array<string, mixed> $quantiles
     */
    #[DataProvider('distributions')]
    public function test_the_upper_quantile_matches_the_certified_value(
        Distribution $distribution,
        array $points,
        array $quantiles,
    ): void {
        /** @var array<string, array{certified: string, scipy: float, attainable: float|null, vanishes?: bool}> $quantiles */
        foreach ($quantiles as $p => $expected) {
            self::assertAgreesWithScipy($distribution->upperQuantile((float) $p), $expected, "isf({$p})");
        }
    }

    /**
     * The quantile must invert the tail it was asked about, all the way down.
     *
     * Round-tripping through the survival function rather than the cumulative
     * is the whole point: survival(upperQuantile(1e-12)) has to come back as
     * 1e-12, which it cannot if either direction went via 1 - the other.
     *
     * @param array<string, mixed> $points
     * @param array<string, mixed> $quantiles
     */
    #[DataProvider('distributions')]
    public function test_the_quantile_and_the_survival_function_invert_each_other(
        Distribution $distribution,
        array $points,
        array $quantiles,
    ): void {
        foreach ([0.5, 0.05, 1.0e-6, 1.0e-12] as $p) {
            $x = $distribution->upperQuantile($p);

            Lre::assertDigits(
                $distribution->survival($x),
                $p,
                sprintf('survival(upperQuantile(%g))', $p),
                11.0,
            );
        }
    }

    /**
     * The two tails must sum to one at every point checked above.
     *
     * Weak on its own -- an implementation computing one from the other
     * satisfies it exactly -- which is why the fixture pins both separately.
     * It is here to catch the opposite error, two independent routes that
     * disagree.
     *
     * @param array<string, mixed> $points
     * @param array<string, mixed> $quantiles
     */
    #[DataProvider('distributions')]
    public function test_the_two_tails_are_complementary(
        Distribution $distribution,
        array $points,
        array $quantiles,
    ): void {
        foreach (array_keys($points) as $x) {
            $value = (float) $x;

            self::assertEqualsWithDelta(
                1.0,
                $distribution->cumulative($value) + $distribution->survival($value),
                1.0e-14,
                "cdf + sf at {$x}",
            );
        }
    }
}
