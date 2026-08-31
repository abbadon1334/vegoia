<?php

declare(strict_types=1);

namespace Vegoia\Tests\Reference\Stats;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Vegoia\Stats\SpecialFunction;
use Vegoia\Tests\Support\Lre;
use Vegoia\Tests\Support\Paths;

/**
 * The special functions, against a certified value and a measured ceiling.
 *
 * PHP has none of these -- no erf, no lgamma, no incomplete gamma or beta --
 * so every one of them is written here, and writing a special function is
 * exactly the kind of work where an implementation can look right on the easy
 * arguments and be worthless on the hard ones. Hence two references rather
 * than one:
 *
 *   * mpmath at 50 digits gives the true value, independent of SciPy;
 *   * SciPy is measured against it, and what SciPy reaches is the ceiling.
 *
 * The requirement is SciPy's accuracy less half a digit, capped at 13. It
 * cannot be met by an implementation that is merely close, and it cannot ask
 * for accuracy a double is unable to hold. Where the true value is zero or
 * smaller than any double, the requirement is to return zero rather than a
 * number of digits.
 *
 * @see tools/generate_special_function_fixtures.py
 */
#[CoversClass(SpecialFunction::class)]
#[Group('reference')]
final class SpecialFunctionTest extends TestCase
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
                (string) file_get_contents(Paths::fixture('stats/special_functions.json')),
                true,
                flags: JSON_THROW_ON_ERROR,
            );
            self::$fixture = $decoded;
        }

        /** @var array<string, mixed> $out */
        $out = self::$fixture[$name];

        return $out;
    }

    /**
     * @param array{certified: string, scipy: float, attainable: float|null, vanishes?: bool} $entry
     */
    private static function assertAgreesWithScipy(float $computed, array $entry, string $what): void
    {
        $certified = (float) $entry['certified'];

        if ($entry['vanishes'] ?? false) {
            self::assertSame(
                0.0,
                $computed,
                "{$what}: the true value is below any double, so the answer must be exactly zero",
            );

            return;
        }

        // A null ceiling means SciPy hit the certified value exactly, so
        // nothing caps us and the standard bar applies.
        $required = $entry['attainable'] === null
            ? (float) Lre::DEFAULT_DIGITS
            : min((float) Lre::DEFAULT_DIGITS, $entry['attainable'] - self::MARGIN);

        Lre::assertDigits($computed, $certified, $what, $required);
    }

    /** @return iterable<string, array{string, array<string, mixed>}> */
    public static function logGammaPoints(): iterable
    {
        /** @var array<string, array<string, mixed>> $entries */
        $entries = self::section('log_gamma');

        foreach ($entries as $x => $entry) {
            yield "lgamma({$x})" => [$x, $entry];
        }
    }

    /** @param array<string, mixed> $entry */
    #[DataProvider('logGammaPoints')]
    public function test_log_gamma_matches_the_certified_value(string $x, array $entry): void
    {
        /** @var array{certified: string, scipy: float, attainable: float|null, vanishes?: bool} $entry */
        self::assertAgreesWithScipy(SpecialFunction::logGamma((float) $x), $entry, "logGamma({$x})");
    }

    /** @return iterable<string, array{string, array{string, array<string, mixed>}}> */
    public static function errorFunctionPoints(): iterable
    {
        /** @var array<string, array<string, array<string, mixed>>> $entries */
        $entries = self::section('error_function');

        foreach ($entries as $x => $pair) {
            foreach ($pair as $which => $entry) {
                yield "{$which}({$x})" => [$x, [$which, $entry]];
            }
        }
    }

    /** @param array{0: string, 1: array<string, mixed>} $case */
    #[DataProvider('errorFunctionPoints')]
    public function test_the_error_function_matches_the_certified_value(string $x, array $case): void
    {
        [$which, $entry] = $case;
        $value = (float) $x;

        /** @var array{certified: string, scipy: float, attainable: float|null, vanishes?: bool} $entry */
        self::assertAgreesWithScipy(
            $which === 'erf' ? SpecialFunction::erf($value) : SpecialFunction::erfc($value),
            $entry,
            "{$which}({$x})",
        );
    }

    /** @return iterable<string, array{array<string, mixed>}> */
    public static function incompleteGammaPoints(): iterable
    {
        /** @var array<string, array<string, mixed>> $entries */
        $entries = self::section('incomplete_gamma');

        foreach ($entries as $key => $entry) {
            yield $key => [$entry];
        }
    }

    /** @param array<string, mixed> $entry */
    #[DataProvider('incompleteGammaPoints')]
    public function test_the_incomplete_gamma_matches_the_certified_value(array $entry): void
    {
        /** @var array{a: float, x: float, P: array<string, mixed>, Q: array<string, mixed>} $entry */
        $a = $entry['a'];
        $x = $entry['x'];

        /** @var array{certified: string, scipy: float, attainable: float|null, vanishes?: bool} $p */
        $p = $entry['P'];
        /** @var array{certified: string, scipy: float, attainable: float|null, vanishes?: bool} $q */
        $q = $entry['Q'];

        self::assertAgreesWithScipy(SpecialFunction::regularizedGammaP($a, $x), $p, "P({$a}, {$x})");
        self::assertAgreesWithScipy(SpecialFunction::regularizedGammaQ($a, $x), $q, "Q({$a}, {$x})");
    }

    /** @return iterable<string, array{array<string, mixed>}> */
    public static function incompleteBetaPoints(): iterable
    {
        /** @var array<string, array<string, mixed>> $entries */
        $entries = self::section('incomplete_beta');

        foreach ($entries as $key => $entry) {
            yield $key => [$entry];
        }
    }

    /** @param array<string, mixed> $entry */
    #[DataProvider('incompleteBetaPoints')]
    public function test_the_incomplete_beta_matches_the_certified_value(array $entry): void
    {
        /** @var array{a: float, b: float, x: float, I: array<string, mixed>} $entry */
        /** @var array{certified: string, scipy: float, attainable: float|null, vanishes?: bool} $i */
        $i = $entry['I'];

        self::assertAgreesWithScipy(
            SpecialFunction::regularizedBeta($entry['x'], $entry['a'], $entry['b']),
            $i,
            "I_{$entry['x']}({$entry['a']}, {$entry['b']})",
        );
    }

    /**
     * P and Q must sum to one, and each must be the accurate one in its own
     * half. A library that computes one as 1 - the other passes this and still
     * loses every digit of the small tail, which is why the fixture pins both
     * separately -- this only checks they are consistent.
     */
    public function test_the_two_gamma_tails_complement_each_other(): void
    {
        foreach ([0.5, 1.0, 5.0, 50.0] as $a) {
            foreach ([0.1, 1.0, 5.0, 20.0] as $x) {
                self::assertEqualsWithDelta(
                    1.0,
                    SpecialFunction::regularizedGammaP($a, $x) + SpecialFunction::regularizedGammaQ($a, $x),
                    1.0e-14,
                    "P + Q at a={$a}, x={$x}",
                );
            }
        }
    }

    /** I_x(a, b) = 1 - I_{1-x}(b, a), the symmetry the continued fraction relies on. */
    public function test_the_incomplete_beta_is_symmetric(): void
    {
        foreach ([[2.0, 3.0], [0.5, 0.5], [10.0, 4.0]] as [$a, $b]) {
            foreach ([0.1, 0.4, 0.5, 0.8] as $x) {
                self::assertEqualsWithDelta(
                    SpecialFunction::regularizedBeta($x, $a, $b),
                    1.0 - SpecialFunction::regularizedBeta(1.0 - $x, $b, $a),
                    1.0e-14,
                    "symmetry at x={$x}, a={$a}, b={$b}",
                );
            }
        }
    }
}
