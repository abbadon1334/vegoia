<?php

declare(strict_types=1);

namespace Vegoia\Stats;

use function array_keys;
use function count;
use function is_nan;
use function min;
use function usort;

use Vegoia\Exception\InvalidArgument;

/**
 * Reading a family of p-values instead of one.
 *
 * A p-value is a claim about one comparison. Twenty of them at the five per
 * cent level is a claim about twenty, and one of those is expected to be wrong
 * by chance alone. This library hands out p-values freely -- one per regression
 * coefficient, one per analysis of variance -- so it also has to hand out the
 * thing that makes a family of them readable, or it is inviting the most
 * common mistake in applied statistics and saying nothing.
 *
 * All three procedures return adjusted p-values rather than a verdict, because
 * an adjusted p-value can be reported, compared and re-thresholded, while a
 * boolean has thrown that away. `rejected()` exists on top for the one thing
 * that would otherwise be left unpinned: whether the comparison at the
 * boundary is `<=` or `<`.
 */
final class MultipleTesting
{
    /**
     * Adjusted p-values, in input order and under the input keys.
     *
     * Keys survive because a caller who named their comparisons should get the
     * names back rather than having to remember what order they passed them
     * in. Sorting happens internally, over a permutation.
     *
     * @param array<array-key, float> $pValues
     *
     * @return array<array-key, float>
     */
    public static function adjust(array $pValues, Adjustment $method): array
    {
        $keys = array_keys($pValues);
        $n = count($keys);

        if ($n === 0) {
            return [];
        }

        foreach ($pValues as $key => $p) {
            // NAN is not hypothetical here: Fit::pValue() in this same library
            // returns it for a zero coefficient with a zero standard error.
            // Handed to usort it produces an arbitrary permutation, and the
            // whole family comes back silently wrong.
            if (is_nan($p)) {
                throw InvalidArgument::notANumber("The p-value at position {$key}");
            }

            if ($p < 0.0 || $p > 1.0) {
                throw InvalidArgument::outOfRange("The p-value at position {$key}", $p, 0.0, 1.0);
            }
        }

        if ($method === Adjustment::Bonferroni) {
            $adjusted = [];

            foreach ($pValues as $key => $p) {
                $adjusted[$key] = min(1.0, $n * $p);
            }

            return $adjusted;
        }

        $order = $keys;
        usort($order, static fn ($a, $b): int => $pValues[$a] <=> $pValues[$b]);

        $adjusted = [];

        if ($method === Adjustment::Holm) {
            // Step-down: walk from the smallest p-value outward, and carry a
            // running maximum. The maximum is what enforces monotonicity --
            // without it the sequence (n - rank) * p can fall as p rises, and
            // a threshold would then reject a larger p-value while refusing a
            // smaller one.
            $running = 0.0;

            foreach ($order as $rank => $key) {
                $running = max($running, ($n - $rank) * $pValues[$key]);
                $adjusted[$key] = min(1.0, $running);
            }
        } else {
            // Step-up: walk from the largest p-value inward with a running
            // minimum. Same argument in the other direction.
            //
            // The ratio is written p / (rank / n) rather than p * n / rank or
            // p * (n / rank). All three are algebraically equal and disagree
            // in the last bit, and because the running minimum propagates, one
            // last-bit difference moves every entry below it. Measured over
            // 200k random triples the three are indistinguishable in accuracy
            // -- around 1.2% correctly rounded each -- so the form is chosen
            // for agreeing with statsmodels bit for bit, which removes a whole
            // category of fixture noise for nothing.
            $running = INF;

            for ($i = $n - 1; $i >= 0; $i--) {
                $key = $order[$i];
                $running = min($running, $pValues[$key] / (($i + 1) / $n));
                $adjusted[$key] = min(1.0, $running);
            }
        }

        // usort visited the family in p-value order, so restore the caller's.
        $inOrder = [];

        foreach ($keys as $key) {
            $inOrder[$key] = $adjusted[$key];
        }

        return $inOrder;
    }

    /**
     * Which members of the family are rejected at this level.
     *
     * A caller could write `$adjusted[$k] <= $alpha` themselves, and that is
     * exactly the reason this exists: the comparison at the boundary is a
     * convention, and a convention left at every call site is a convention
     * nobody pinned. It is `<=`, matching statsmodels.
     *
     * @param array<array-key, float> $pValues
     *
     * @return array<array-key, bool>
     */
    public static function rejected(
        array $pValues,
        Adjustment $method,
        float $alpha = 0.05,
    ): array {
        if (! ($alpha > 0.0 && $alpha <= 1.0)) {
            throw InvalidArgument::outOfRange('A significance level', $alpha, 0.0, 1.0);
        }

        $rejected = [];

        foreach (self::adjust($pValues, $method) as $key => $adjusted) {
            $rejected[$key] = $adjusted <= $alpha;
        }

        return $rejected;
    }
}
