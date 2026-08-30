<?php

declare(strict_types=1);

namespace Vegoia\Graph\Community;

use function count;
use function log;
use function max;
use function min;

use Vegoia\Exception\InvalidArgument;
use Vegoia\Graph\Partition;
use Vegoia\Support\CompensatedSum;

/**
 * How much two partitions of the same nodes agree.
 *
 * Needed for two jobs that look different and are the same measurement.
 * Evaluating community detection against a known ground truth is one. The
 * other is comparing one run to the next: community labels are arbitrary and
 * get renumbered every time, so "did the partition change between rounds"
 * cannot be answered by comparing labels, and comparing sets by hand degrades
 * into ad-hoc overlap heuristics.
 *
 * Three measures, because they answer different questions:
 *
 *   * NMI is information-theoretic -- how much knowing one partition tells you
 *     about the other. Insensitive to community sizes, always in [0, 1].
 *   * ARI counts agreeing pairs of nodes, corrected for chance. It can go
 *     negative when two partitions agree less than random labelling would,
 *     which NMI cannot express.
 *   * VI is a true metric: it obeys the triangle inequality, so distances
 *     between partitions can be built from it. NMI and ARI cannot do this.
 *
 * NMI's normalisation is not universal -- arithmetic, geometric, min and max
 * of the two entropies are all in use and disagree. This follows
 * scikit-learn's default, the arithmetic mean, so results are comparable with
 * what the Python ecosystem reports.
 */
final class Agreement
{
    /**
     * NMI = 2 * I(a; b) / (H(a) + H(b)).
     *
     * Returns 1.0 when both entropies vanish -- two partitions that each put
     * everything in one community carry no information and cannot disagree.
     * Returning 0.0 there, which the raw formula's 0/0 invites, would report
     * two identical partitions as maximally different.
     */
    public static function normalisedMutualInformation(Partition $a, Partition $b): float
    {
        [$sizesA, $sizesB, $joint, $n] = self::contingency($a, $b);

        if ($n === 0) {
            return 1.0;
        }

        $entropyA = self::entropy($sizesA, $n);
        $entropyB = self::entropy($sizesB, $n);
        $total = $entropyA + $entropyB;

        if ($total === 0.0) {
            return 1.0;
        }

        $mutual = new CompensatedSum();

        foreach ($joint as $left => $row) {
            foreach ($row as $right => $shared) {
                // p(i,j) * log( p(i,j) / (p(i) p(j)) ), rearranged so the
                // logarithm takes a ratio of counts rather than of tiny
                // probabilities, which keeps its argument away from underflow.
                $mutual->add(
                    ($shared / $n) * log(($shared * $n) / ($sizesA[$left] * $sizesB[$right]))
                );
            }
        }

        // Rounding can push a perfect match a hair above 1.
        return max(0.0, min(1.0, 2.0 * $mutual->value() / $total));
    }

    /**
     * The Rand index counts pairs of nodes the two partitions treat alike --
     * together in both, or apart in both. Uncorrected it is badly behaved:
     * two random partitions of a large graph score near 1, because almost
     * every pair is apart in both. ARI subtracts the score chance alone would
     * produce, which places 0 at "no better than random" and lets genuine
     * disagreement go negative.
     */
    public static function adjustedRandIndex(Partition $a, Partition $b): float
    {
        [$sizesA, $sizesB, $joint, $n] = self::contingency($a, $b);

        if ($n < 2) {
            return 1.0;
        }

        $agreeing = 0.0;

        foreach ($joint as $row) {
            foreach ($row as $shared) {
                $agreeing += self::pairs($shared);
            }
        }

        $withinA = 0.0;
        foreach ($sizesA as $size) {
            $withinA += self::pairs($size);
        }

        $withinB = 0.0;
        foreach ($sizesB as $size) {
            $withinB += self::pairs($size);
        }

        $allPairs = self::pairs($n);
        $expected = $withinA * $withinB / $allPairs;
        $maximum = 0.5 * ($withinA + $withinB);

        if ($maximum === $expected) {
            // Both partitions are degenerate in the same way (all singletons,
            // or all one community): they agree perfectly and the correction
            // has nothing to correct.
            return 1.0;
        }

        return ($agreeing - $expected) / ($maximum - $expected);
    }

    /**
     * VI = H(a) + H(b) - 2 I(a; b), in nats.
     *
     * Zero exactly when the partitions are identical, and unlike NMI and ARI
     * it satisfies the triangle inequality, so it is the one to use when a
     * distance is actually needed -- clustering partitions, or tracking drift
     * across a sequence of runs.
     */
    public static function variationOfInformation(Partition $a, Partition $b): float
    {
        [$sizesA, $sizesB, $joint, $n] = self::contingency($a, $b);

        if ($n === 0) {
            return 0.0;
        }

        $mutual = new CompensatedSum();

        foreach ($joint as $left => $row) {
            foreach ($row as $right => $shared) {
                $mutual->add(
                    ($shared / $n) * log(($shared * $n) / ($sizesA[$left] * $sizesB[$right]))
                );
            }
        }

        $value = self::entropy($sizesA, $n) + self::entropy($sizesB, $n) - 2.0 * $mutual->value();

        // Cancellation can leave a tiny negative where the true value is zero.
        return max(0.0, $value);
    }

    /**
     * @return array{list<int>, list<int>, array<int, array<int, int>>, int}
     */
    private static function contingency(Partition $a, Partition $b): array
    {
        $left = $a->membership();
        $right = $b->membership();

        if (count($left) !== count($right)) {
            throw InvalidArgument::malformedEdge(
                'Partitions cover different numbers of nodes: ' . count($left) . ' and ' . count($right)
            );
        }

        $sizesA = [];
        $sizesB = [];
        $joint = [];

        foreach ($left as $node => $community) {
            $other = $right[$node];

            $sizesA[$community] = ($sizesA[$community] ?? 0) + 1;
            $sizesB[$other] = ($sizesB[$other] ?? 0) + 1;
            $joint[$community][$other] = ($joint[$community][$other] ?? 0) + 1;
        }

        /** @var list<int> $sizesA */
        /** @var list<int> $sizesB */
        return [$sizesA, $sizesB, $joint, count($left)];
    }

    /** @param list<int> $sizes */
    private static function entropy(array $sizes, int $n): float
    {
        $entropy = new CompensatedSum();

        foreach ($sizes as $size) {
            if ($size > 0) {
                $probability = $size / $n;
                $entropy->add(-$probability * log($probability));
            }
        }

        return $entropy->value();
    }

    /** Pairs that can be drawn from a group: n(n-1)/2, as a float to avoid overflow. */
    private static function pairs(int $size): float
    {
        return $size * ($size - 1) / 2.0;
    }
}
