<?php

declare(strict_types=1);

namespace Vegoia\Stats;

use function array_keys;
use function array_values;
use function count;
use function ksort;
use function max;
use function min;
use function range;
use function sqrt;
use function usort;

use Vegoia\Exception\InvalidArgument;
use Vegoia\Support\CompensatedSum;

/**
 * Correlation between two samples, three ways.
 *
 * They measure different things and choosing between them is not a matter of
 * taste. Anscombe's quartet makes the point: four datasets with Pearson r of
 * 0.816 apiece, one a clean linear relationship, one a parabola, one a line
 * with an outlier, one a vertical stack with a single distant point. Pearson
 * cannot tell them apart. Spearman scores them 0.82, 0.69 and 0.50.
 *
 *   * Pearson measures LINEAR association only. A perfect parabola scores 0.
 *   * Spearman is Pearson on the ranks, so it detects any monotone
 *     relationship and is unmoved by how far out an outlier lies -- only by
 *     the fact that it is last.
 *   * Kendall counts concordant against discordant pairs. Similar in spirit to
 *     Spearman but with a directly interpretable meaning -- the difference in
 *     probability that two observations agree in order versus disagree -- and
 *     better behaved on small samples.
 *
 * Ties are where implementations quietly diverge, so they are handled
 * explicitly: Spearman uses midranks, and Kendall uses the tau-b correction,
 * which is SciPy's default and the only variant that reaches 1 on a perfectly
 * ordered sample containing ties.
 */
final class Correlation
{
    /**
     * Pearson's r, computed from deviations rather than from raw sums.
     *
     * The shortcut form -- (n*sum(xy) - sum(x)sum(y)) over a matching
     * denominator -- is what most implementations use and it fails the same
     * way the naive variance does: on large values with small spread every
     * digit cancels. The `large_values` fixture, four numbers just above 1e8,
     * returns garbage from it.
     *
     * @param list<float> $x
     * @param list<float> $y
     */
    public static function pearson(array $x, array $y): float
    {
        $n = self::assertPaired($x, $y);

        $meanX = Descriptive::of($x)->mean();
        $meanY = Descriptive::of($y)->mean();

        $covariance = new CompensatedSum();
        $varianceX = new CompensatedSum();
        $varianceY = new CompensatedSum();

        for ($i = 0; $i < $n; $i++) {
            $dx = $x[$i] - $meanX;
            $dy = $y[$i] - $meanY;

            $covariance->add($dx * $dy);
            $varianceX->add($dx * $dx);
            $varianceY->add($dy * $dy);
        }

        $denominator = sqrt($varianceX->value()) * sqrt($varianceY->value());

        if ($denominator === 0.0) {
            throw InvalidArgument::malformedEdge(
                'Correlation is undefined when a sample has no variation'
            );
        }

        // Rounding can carry a perfect correlation a hair past 1.
        return max(-1.0, min(1.0, $covariance->value() / $denominator));
    }

    /**
     * Spearman's rho: Pearson applied to midranks.
     *
     * @param list<float> $x
     * @param list<float> $y
     */
    public static function spearman(array $x, array $y): float
    {
        self::assertPaired($x, $y);

        return self::pearson(self::midranks($x), self::midranks($y));
    }

    /**
     * Kendall's tau-b.
     *
     *     tau = (C - D) / sqrt((n0 - n1) (n0 - n2))
     *
     * with C and D the concordant and discordant pairs, n0 all pairs, and n1,
     * n2 the pairs tied within each sample. Without the tie correction (tau-a)
     * a perfectly ordered sample containing ties cannot reach 1, which makes
     * the coefficient hard to interpret exactly where ties are common.
     *
     * @param list<float> $x
     * @param list<float> $y
     */
    public static function kendall(array $x, array $y): float
    {
        $n = self::assertPaired($x, $y);

        $concordant = 0;
        $discordant = 0;

        // O(n^2). Knight's algorithm gets this to O(n log n) via merge sort,
        // which matters from tens of thousands of points; below that the
        // simple form is faster in PHP and obviously correct.
        for ($i = 0; $i < $n - 1; $i++) {
            for ($j = $i + 1; $j < $n; $j++) {
                $dx = $x[$i] <=> $x[$j];
                $dy = $y[$i] <=> $y[$j];
                $sign = $dx * $dy;

                if ($sign > 0) {
                    $concordant++;
                } elseif ($sign < 0) {
                    $discordant++;
                }
                // A pair tied in either sample is neither, and is accounted
                // for by the correction below.
            }
        }

        $allPairs = $n * ($n - 1) / 2.0;
        $tiedX = self::tiedPairs($x);
        $tiedY = self::tiedPairs($y);

        $denominator = sqrt(($allPairs - $tiedX) * ($allPairs - $tiedY));

        if ($denominator === 0.0) {
            throw InvalidArgument::malformedEdge(
                'Kendall tau is undefined when a sample is entirely tied'
            );
        }

        return max(-1.0, min(1.0, ($concordant - $discordant) / $denominator));
    }

    /**
     * Pairwise correlation matrix over named columns.
     *
     * Symmetric with ones on the diagonal, computed once per pair rather than
     * twice.
     *
     * @param  array<string, list<float>> $columns
     * @return array<string, array<string, float>>
     */
    public static function matrix(array $columns, string $method = 'pearson'): array
    {
        $names = array_keys($columns);
        $matrix = [];

        foreach ($names as $row) {
            foreach ($names as $column) {
                if ($row === $column) {
                    $matrix[$row][$column] = 1.0;

                    continue;
                }

                if (isset($matrix[$column][$row])) {
                    $matrix[$row][$column] = $matrix[$column][$row];

                    continue;
                }

                $matrix[$row][$column] = match ($method) {
                    'pearson' => self::pearson($columns[$row], $columns[$column]),
                    'spearman' => self::spearman($columns[$row], $columns[$column]),
                    'kendall' => self::kendall($columns[$row], $columns[$column]),
                    default => throw InvalidArgument::malformedEdge(
                        "Unknown correlation method '{$method}'; expected pearson, spearman or kendall"
                    ),
                };
            }
        }

        return $matrix;
    }

    /**
     * Ranks with ties averaged -- the midrank convention. Three values tied
     * for positions 2, 3 and 4 all become 3.0, which keeps the rank sum equal
     * to what distinct values would have produced.
     *
     * @param  list<float> $values
     * @return list<float>
     */
    private static function midranks(array $values): array
    {
        $n = count($values);
        $order = range(0, $n - 1);

        usort($order, static fn (int $a, int $b): int => $values[$a] <=> $values[$b]);

        $ranks = [];

        for ($i = 0; $i < $n;) {
            $j = $i;

            while ($j + 1 < $n && $values[$order[$j + 1]] === $values[$order[$i]]) {
                $j++;
            }

            // Positions i..j are tied; ranks are 1-based, so their mean is
            // (i + j) / 2 + 1.
            $rank = ($i + $j) / 2.0 + 1.0;

            for ($k = $i; $k <= $j; $k++) {
                $ranks[$order[$k]] = $rank;
            }

            $i = $j + 1;
        }

        ksort($ranks);

        /** @var list<float> $ranks */
        return array_values($ranks);
    }

    /**
     * Pairs tied within a sample: sum of t(t-1)/2 over each group of equal
     * values.
     *
     * @param list<float> $values
     */
    private static function tiedPairs(array $values): float
    {
        $counts = [];

        foreach ($values as $value) {
            $key = (string) $value;
            $counts[$key] = ($counts[$key] ?? 0) + 1;
        }

        $tied = 0.0;

        foreach ($counts as $count) {
            $tied += $count * ($count - 1) / 2.0;
        }

        return $tied;
    }

    /**
     * @param list<float> $x
     * @param list<float> $y
     */
    private static function assertPaired(array $x, array $y): int
    {
        $n = count($x);

        if ($n !== count($y)) {
            throw InvalidArgument::malformedEdge(
                "Correlation needs paired samples: {$n} and " . count($y) . ' values given'
            );
        }

        if ($n < 2) {
            throw InvalidArgument::tooFewValues('Correlation', $n, 2);
        }

        return $n;
    }
}
