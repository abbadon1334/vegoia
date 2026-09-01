<?php

declare(strict_types=1);

namespace Vegoia\Stats;

use function array_values;
use function count;
use function ksort;
use function range;
use function usort;

/**
 * Ranks, and what ties do to them.
 *
 * Extracted because three things need to agree about what a tie is, and two of
 * them did not. Kendall's tau-b counts tied pairs in its denominator and
 * ordered pairs in its numerator; Mann-Whitney and Kruskal-Wallis correct
 * their variance and their statistic by the size of each tied group. If any
 * two of those disagree the answer is not merely imprecise, it is incoherent
 * -- a number that is neither what you get with the tie nor what you get
 * without it.
 *
 * That is not hypothetical. Correlation used to decide ties for its
 * denominator by casting each value to a string, and PHP's default `precision`
 * of 14 makes distinct doubles collide:
 *
 *     (string) 0.1 === (string) 0.10000000000000012   // true
 *     0.1 === 0.10000000000000012                     // false
 *
 * so on ordinary data -- 0.1 and a few units in the last place above it --
 * tau-b returned 0.759, where the answer is 0.733 and the answer had the pair
 * genuinely been tied would have been 0.690. The numerator counted the pair as
 * ordered and the denominator corrected it as tied.
 *
 * Everything here compares values with `===`, on a sorted copy. Two doubles
 * are tied when they are the same double, which is the only definition that
 * cannot drift.
 */
final class Ranks
{
    /**
     * Ranks with ties averaged: the midrank convention.
     *
     * Three values sharing positions 2, 3 and 4 all become 3.0, which keeps
     * the total of the ranks equal to what distinct values would have given --
     * n(n+1)/2 either way. Every rank statistic downstream relies on that,
     * because it is what lets the null mean of a rank sum stay in closed form.
     *
     * @param list<float> $values
     *
     * @return list<float> parallel to the input, not to the sorted order
     */
    public static function midranks(array $values): array
    {
        $n = count($values);

        // range(0, -1) counts downwards and hands back [0, -1], which is not
        // an empty list and not a warning-free way to ask for one. Correlation
        // never reached this because it refuses samples below two long before
        // ranking them; this is a public primitive and has no such guard.
        if ($n === 0) {
            return [];
        }

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
     * The size of each group of equal values, in ascending order of value.
     *
     * Groups of one are included, so the sizes always sum to the sample size.
     * Callers that only want genuine ties filter for size above one; callers
     * that want the tie-correction sums want all of them, because the formulae
     * are written over every group and a group of one contributes zero anyway.
     *
     * Found by sorting and walking, not by grouping on a key. A hash key has
     * to be a string, and turning a float into a string is where the defect
     * this class exists to prevent came from.
     *
     * @param list<float> $values
     *
     * @return list<int>
     */
    public static function tieSizes(array $values): array
    {
        $sorted = $values;
        sort($sorted);

        $sizes = [];
        $n = count($sorted);

        for ($i = 0; $i < $n;) {
            $j = $i;

            while ($j + 1 < $n && $sorted[$j + 1] === $sorted[$i]) {
                $j++;
            }

            $sizes[] = $j - $i + 1;
            $i = $j + 1;
        }

        return $sizes;
    }

    /**
     * The number of pairs that are tied, summed over the groups: sum of
     * t(t-1)/2.
     *
     * The quantity Kendall's tau-b subtracts from its denominator. Accumulated
     * as an integer for as long as it can be: a group of t tied values
     * contributes about t^2/2, which stays exact in a 64-bit integer to
     * t around 1.3e8 and would have gone approximate in a double at 1.3e8 as
     * well -- but the sum over many groups is what benefits, and it costs
     * nothing to be exact until the last step.
     *
     * @param list<float> $values
     */
    public static function tiedPairs(array $values): float
    {
        $tied = 0;

        foreach (self::tieSizes($values) as $size) {
            $tied += intdiv($size * ($size - 1), 2);
        }

        return (float) $tied;
    }
}
