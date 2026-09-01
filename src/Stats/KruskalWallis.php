<?php

declare(strict_types=1);

namespace Vegoia\Stats;

use function array_filter;
use function array_slice;
use function array_values;
use function count;

use Vegoia\Exception\InvalidArgument;
use Vegoia\Stats\Distribution\ChiSquared;
use Vegoia\Support\CompensatedSum;

/**
 * One-way analysis of variance, on the ranks.
 *
 * The same question OneWayAnova asks -- do these groups differ? -- without
 * assuming the groups are normal or share a variance. It replaces every
 * observation with its rank among all of them and compares the groups' mean
 * ranks, so what it costs is sensitivity to how far apart the groups are and
 * what it buys is not caring what shape they have.
 *
 * `grouped()` mirrors OneWayAnova::grouped() deliberately, message and all:
 * the two are alternatives for the same job and should be swappable at the
 * call site.
 */
final readonly class KruskalWallis
{
    private function __construct(
        public float $statistic,
        public float $uncorrectedStatistic,
        public float $tieCorrection,
        public int $degreesOfFreedom,
        public int $groups,
        public int $observations,
    ) {
    }

    /** @param list<list<float>> $groups one list of observations per group */
    public static function of(array $groups): self
    {
        // Empty groups are dropped rather than refused, exactly as
        // OneWayAnova::of() drops them.
        $groups = array_values(array_filter($groups, static fn (array $g): bool => $g !== []));
        $count = count($groups);

        if ($count < 2) {
            throw InvalidArgument::tooFewValues('Kruskal-Wallis', $count, 2);
        }

        $pooled = [];
        $sizes = [];

        foreach ($groups as $group) {
            $sizes[] = count($group);

            foreach ($group as $value) {
                $pooled[] = $value;
            }
        }

        $n = count($pooled);

        if ($n <= $count) {
            throw InvalidArgument::tooFewValues(
                'Kruskal-Wallis needs more observations than groups',
                $n,
                $count + 1,
            );
        }

        $ranks = Ranks::midranks($pooled);
        $grandMean = ($n + 1) / 2.0;

        // The centred form, not the textbook
        //     12/(N(N+1)) * sum(R_i^2/n_i) - 3(N+1).
        // The two are algebraically identical and the textbook one differences
        // 3(N+1) from a quantity of the same size while H itself stays O(1):
        // measured against exact rationals, it holds 11.95 digits at N = 60000
        // where this form holds 13.54, and the gap widens with N. OneWayAnova
        // makes the same move for the same reason.
        $between = new CompensatedSum();
        $offset = 0;

        foreach ($sizes as $size) {
            $mean = new CompensatedSum();

            foreach (array_slice($ranks, $offset, $size) as $rank) {
                $mean->add($rank);
            }

            $deviation = $mean->dividedBy((float) $size) - $grandMean;
            $between->add($size * $deviation * $deviation);
            $offset += $size;
        }

        $uncorrected = 12.0 / ($n * ($n + 1)) * $between->value();

        // Divided by the tie factor, which is what SciPy reports as H. The
        // uncorrected value is kept beside it because a reader checking
        // against a textbook will find that number and should see the library
        // knows about it.
        $tied = 0;

        foreach (Ranks::tieSizes($pooled) as $size) {
            $tied += $size * $size * $size - $size;
        }

        $tieCorrection = 1.0 - $tied / ((float) $n * $n * $n - $n);

        if ($tieCorrection <= 0.0) {
            throw InvalidArgument::malformedEdge(
                'Kruskal-Wallis is undefined when every observation is tied: the ranks carry no '
                . 'information about the groups, so there is nothing to divide'
            );
        }

        return new self(
            $uncorrected / $tieCorrection,
            $uncorrected,
            $tieCorrection,
            $count - 1,
            $count,
            $n,
        );
    }

    /**
     * @param list<float>      $values
     * @param list<int|string> $labels one per value
     */
    public static function grouped(array $values, array $labels): self
    {
        if (count($values) !== count($labels)) {
            throw InvalidArgument::malformedEdge(
                'Kruskal-Wallis needs one label per observation: ' . count($values)
                . ' values and ' . count($labels) . ' labels'
            );
        }

        $groups = [];

        foreach ($values as $index => $value) {
            $groups[$labels[$index]][] = $value;
        }

        /** @var list<list<float>> $lists */
        $lists = array_values($groups);

        return self::of($lists);
    }

    /**
     * How often groups this different in rank arise when they are all the
     * same.
     *
     * The upper tail of a chi-squared, computed directly: only a large H is
     * evidence, so the test is one-sided by construction.
     */
    public function pValue(): float
    {
        return new ChiSquared((float) $this->degreesOfFreedom)->survival($this->statistic);
    }
}
