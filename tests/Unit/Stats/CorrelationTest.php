<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit\Stats;

use function file_get_contents;
use function json_decode;

use const JSON_THROW_ON_ERROR;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Stats\Correlation;
use Vegoia\Tests\Support\Paths;

/**
 * Correlation against SciPy.
 *
 * The cases include Anscombe's quartet, which exists to show that a
 * correlation coefficient alone tells you almost nothing: three of its four
 * datasets have Pearson r within 0.0003 of each other while being a clean
 * line, a parabola, and a vertical stack with one distant point. Spearman
 * separates them at 0.82, 0.69 and 0.50. Having all three measures agree with
 * an independent implementation on those inputs is worth more than any number
 * of well-behaved ones.
 *
 * Ties are the other reason for these fixtures: Spearman needs midranks and
 * Kendall has three tie corrections in circulation. SciPy's tau-b is pinned.
 */
#[CoversClass(Correlation::class)]
final class CorrelationTest extends TestCase
{
    /** @return iterable<string, array{string}> */
    public static function cases(): iterable
    {
        foreach (self::fixtures() as $name => $_) {
            yield $name => [$name];
        }
    }

    #[DataProvider('cases')]
    public function test_pearson_matches_scipy(string $case): void
    {
        $entry = self::fixtures()[$case];

        self::assertEqualsWithDelta(
            $entry['pearson'],
            Correlation::pearson($entry['x'], $entry['y']),
            1.0e-12,
            "{$case}: Pearson r",
        );
    }

    #[DataProvider('cases')]
    public function test_spearman_matches_scipy(string $case): void
    {
        $entry = self::fixtures()[$case];

        self::assertEqualsWithDelta(
            $entry['spearman'],
            Correlation::spearman($entry['x'], $entry['y']),
            1.0e-12,
            "{$case}: Spearman rho",
        );
    }

    #[DataProvider('cases')]
    public function test_kendall_matches_scipy(string $case): void
    {
        $entry = self::fixtures()[$case];

        self::assertEqualsWithDelta(
            $entry['kendall'],
            Correlation::kendall($entry['x'], $entry['y']),
            1.0e-12,
            "{$case}: Kendall tau-b",
        );
    }

    /**
     * The three measure different things, and Anscombe is the demonstration
     * that this matters rather than a technicality.
     */
    public function test_the_three_measures_disagree_where_they_should(): void
    {
        $fixtures = self::fixtures();

        // Nearly identical Pearson across three very different datasets.
        $r = [];
        foreach (['anscombe_1', 'anscombe_2', 'anscombe_4'] as $case) {
            $r[] = Correlation::pearson($fixtures[$case]['x'], $fixtures[$case]['y']);
        }

        self::assertEqualsWithDelta($r[0], $r[1], 1.0e-3, 'Pearson cannot tell them apart');
        self::assertEqualsWithDelta($r[0], $r[2], 1.0e-3);

        // Spearman can.
        $rho = [];
        foreach (['anscombe_1', 'anscombe_2', 'anscombe_4'] as $case) {
            $rho[] = Correlation::spearman($fixtures[$case]['x'], $fixtures[$case]['y']);
        }

        self::assertGreaterThan(0.1, abs($rho[0] - $rho[1]));
        self::assertGreaterThan(0.1, abs($rho[1] - $rho[2]));
    }

    /** A perfect parabola is perfectly monotone in neither direction. */
    public function test_pearson_is_blind_to_a_non_linear_relationship(): void
    {
        self::assertEqualsWithDelta(0.0, Correlation::pearson([-2.0, -1.0, 0.0, 1.0, 2.0], [4.0, 1.0, 0.0, 1.0, 4.0]), 1.0e-15);
    }

    /** Squaring is monotone, so the rank-based measures see it perfectly. */
    public function test_rank_measures_see_a_monotone_relationship_pearson_understates(): void
    {
        $x = [1.0, 2.0, 3.0, 4.0, 5.0];
        $y = [1.0, 4.0, 9.0, 16.0, 25.0];

        self::assertLessThan(0.99, Correlation::pearson($x, $y));

        // Perfect in exact arithmetic; the last bit is not reachable through
        // the rank transform and the division that follows it.
        self::assertEqualsWithDelta(1.0, Correlation::spearman($x, $y), 1.0e-15);
        self::assertEqualsWithDelta(1.0, Correlation::kendall($x, $y), 1.0e-15);
    }

    /**
     * The shortcut formula -- n*sum(xy) - sum(x)sum(y) over a matching
     * denominator -- returns nonsense here, the same way the naive variance
     * does on NIST's NumAcc sets.
     */
    public function test_it_survives_large_values_with_small_spread(): void
    {
        self::assertEqualsWithDelta(
            1.0,
            Correlation::pearson([1.0e8 + 1, 1.0e8 + 2, 1.0e8 + 3, 1.0e8 + 4], [1.0e8 + 2, 1.0e8 + 4, 1.0e8 + 6, 1.0e8 + 8]),
            1.0e-12,
        );
    }

    public function test_the_matrix_is_symmetric_with_ones_on_the_diagonal(): void
    {
        $matrix = Correlation::matrix([
            'a' => [1.0, 2.0, 3.0, 4.0],
            'b' => [2.0, 4.0, 6.0, 8.0],
            'c' => [4.0, 3.0, 2.0, 1.0],
        ]);

        self::assertSame(1.0, $matrix['a']['a'], 'the diagonal is set, not computed');
        self::assertEqualsWithDelta(1.0, $matrix['a']['b'], 1.0e-15, 'a and b are the same line');
        self::assertEqualsWithDelta(-1.0, $matrix['a']['c'], 1.0e-15);
        self::assertSame($matrix['a']['c'], $matrix['c']['a'], 'computed once, mirrored');
    }

    public function test_the_matrix_accepts_the_rank_based_methods_too(): void
    {
        $columns = ['a' => [1.0, 2.0, 3.0, 4.0], 'b' => [1.0, 4.0, 9.0, 16.0]];

        self::assertEqualsWithDelta(1.0, Correlation::matrix($columns, 'spearman')['a']['b'], 1.0e-15);
        self::assertEqualsWithDelta(1.0, Correlation::matrix($columns, 'kendall')['a']['b'], 1.0e-15);
        self::assertLessThan(1.0, Correlation::matrix($columns, 'pearson')['a']['b']);
    }

    public function test_it_refuses_an_unknown_method(): void
    {
        $this->expectException(InvalidArgument::class);

        Correlation::matrix(['a' => [1.0, 2.0], 'b' => [2.0, 1.0]], 'cosine');
    }

    public function test_it_refuses_unpaired_or_constant_samples(): void
    {
        $this->expectException(InvalidArgument::class);

        Correlation::pearson([1.0, 2.0, 3.0], [1.0, 2.0]);
    }

    public function test_a_sample_with_no_variation_has_no_correlation(): void
    {
        $this->expectException(InvalidArgument::class);

        Correlation::pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]);
    }

    public function test_it_refuses_a_single_observation(): void
    {
        $this->expectException(InvalidArgument::class);

        Correlation::pearson([1.0], [2.0]);
    }

    /** @return array<string, array{x: list<float>, y: list<float>, pearson: float, spearman: float, kendall: float}> */
    private static function fixtures(): array
    {
        /** @var array{cases: array<string, array{x: list<float>, y: list<float>, pearson: float, spearman: float, kendall: float}>} $data */
        $data = json_decode(
            (string) file_get_contents(Paths::fixture('correlation.json')),
            associative: true,
            flags: JSON_THROW_ON_ERROR,
        );

        return $data['cases'];
    }
}
