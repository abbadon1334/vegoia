<?php

declare(strict_types=1);

namespace Vegoia\Tests\Reference\Stats;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Vegoia\Stats\Adjustment;
use Vegoia\Stats\MultipleTesting;
use Vegoia\Tests\Support\Lre;
use Vegoia\Tests\Support\Paths;

/**
 * Multiple testing correction, against statsmodels.
 *
 * Ten families, chosen for the shapes that separate the three procedures
 * rather than for variety: an unsorted input, because nothing says a caller's
 * p-values arrive in order; a family whose raw Benjamini-Hochberg ratios are
 * decreasing, where the step-up is the difference between rejecting the right
 * hypothesis and the wrong one; ties; and the boundary values 0 and 1.
 *
 * @see tools/generate_hypothesis_fixtures.py
 */
#[CoversClass(MultipleTesting::class)]
#[CoversClass(Adjustment::class)]
#[Group('reference')]
final class MultipleTestingTest extends TestCase
{
    /** @return iterable<string, array{string, array<string, mixed>, float}> */
    public static function families(): iterable
    {
        /** @var array{alpha: float, multiple_testing: array<string, array<string, mixed>>} $decoded */
        $decoded = json_decode(
            (string) file_get_contents(Paths::fixture('stats/hypothesis.json')),
            true,
            flags: JSON_THROW_ON_ERROR,
        );

        foreach ($decoded['multiple_testing'] as $name => $entry) {
            yield $name => [$name, $entry, $decoded['alpha']];
        }
    }

    /** @return array<string, Adjustment> */
    private static function methods(): array
    {
        return [
            'bonferroni' => Adjustment::Bonferroni,
            'holm' => Adjustment::Holm,
            'benjamini_hochberg' => Adjustment::BenjaminiHochberg,
        ];
    }

    /** @param array<string, mixed> $entry */
    #[DataProvider('families')]
    public function test_the_adjusted_values_match_statsmodels(
        string $name,
        array $entry,
        float $alpha,
    ): void {
        /** @var array{p_values: list<float>} $family */
        $family = $entry;

        foreach (self::methods() as $key => $method) {
            /** @var array{adjusted: list<float>} $expected */
            $expected = $entry[$key];

            $adjusted = MultipleTesting::adjust($family['p_values'], $method);

            self::assertSame(
                array_keys($family['p_values']),
                array_keys($adjusted),
                "{$name}: {$key} did not return the family in input order",
            );

            foreach ($expected['adjusted'] as $index => $value) {
                Lre::assertDigits(
                    $adjusted[$index],
                    $value,
                    "{$name}: {$key} at position {$index}",
                    14.0,
                );
            }
        }
    }

    /** @param array<string, mixed> $entry */
    #[DataProvider('families')]
    public function test_the_rejections_match_statsmodels(
        string $name,
        array $entry,
        float $alpha,
    ): void {
        /** @var array{p_values: list<float>} $family */
        $family = $entry;

        foreach (self::methods() as $key => $method) {
            /** @var array{rejected: list<bool>} $expected */
            $expected = $entry[$key];

            self::assertSame(
                $expected['rejected'],
                array_values(MultipleTesting::rejected($family['p_values'], $method, $alpha)),
                "{$name}: {$key}",
            );
        }
    }

    /**
     * Adjusted p-values never decrease as the raw ones increase.
     *
     * The property both step procedures exist to enforce, and the one a
     * caller relies on without knowing it: thresholding an adjusted family
     * has to reject a prefix of it. Without the monotone pass the raw
     * Benjamini-Hochberg ratios on `ratios_out_of_order` are 0.08 and 0.041,
     * so a threshold would reject the larger p-value and not the smaller.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('families')]
    public function test_the_adjusted_family_is_monotone(string $name, array $entry, float $alpha): void
    {
        /** @var array{p_values: list<float>} $entry */
        $raw = $entry['p_values'];

        foreach (self::methods() as $key => $method) {
            $adjusted = MultipleTesting::adjust($raw, $method);
            $order = array_keys($raw);

            usort($order, static fn (int $a, int $b): int => $raw[$a] <=> $raw[$b]);

            $previous = -INF;

            foreach ($order as $index) {
                self::assertGreaterThanOrEqual(
                    $previous,
                    $adjusted[$index],
                    "{$name}: {$key} is not monotone in the raw p-value",
                );
                $previous = $adjusted[$index];
            }
        }
    }

    /**
     * Holm rejects everything Bonferroni does, and sometimes more, at the same
     * guarantee. That is the whole reason to prefer it, and it is checkable
     * rather than a claim.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('families')]
    public function test_holm_is_never_weaker_than_bonferroni(
        string $name,
        array $entry,
        float $alpha,
    ): void {
        /** @var array{p_values: list<float>} $entry */
        $bonferroni = MultipleTesting::adjust($entry['p_values'], Adjustment::Bonferroni);
        $holm = MultipleTesting::adjust($entry['p_values'], Adjustment::Holm);

        foreach ($bonferroni as $index => $value) {
            self::assertLessThanOrEqual($value, $holm[$index], "{$name}: at position {$index}");
        }
    }

    /**
     * Controlling the false discovery rate is a weaker demand than controlling
     * the family-wise error rate, so Benjamini-Hochberg never adjusts a
     * p-value further than Holm does.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('families')]
    public function test_benjamini_hochberg_is_never_stricter_than_holm(
        string $name,
        array $entry,
        float $alpha,
    ): void {
        /** @var array{p_values: list<float>} $entry */
        $holm = MultipleTesting::adjust($entry['p_values'], Adjustment::Holm);
        $bh = MultipleTesting::adjust($entry['p_values'], Adjustment::BenjaminiHochberg);

        foreach ($holm as $index => $value) {
            self::assertLessThanOrEqual($value, $bh[$index], "{$name}: at position {$index}");
        }
    }

    /**
     * Keys survive, so a caller who named their comparisons gets the names
     * back rather than having to remember the order they passed them in.
     */
    public function test_string_keys_are_preserved(): void
    {
        $family = ['ab' => 0.01, 'ac' => 0.2, 'bc' => 0.03];

        foreach (self::methods() as $method) {
            $adjusted = MultipleTesting::adjust($family, $method);

            self::assertSame(['ab', 'ac', 'bc'], array_keys($adjusted));
            self::assertGreaterThan($adjusted['ab'], $adjusted['ac']);
        }
    }
}
