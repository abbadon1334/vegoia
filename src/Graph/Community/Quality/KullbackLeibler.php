<?php

declare(strict_types=1);

namespace Vegoia\Graph\Community\Quality;

use function log;

/**
 * Binary Kullback-Leibler divergence, shared by Surprise and Significance.
 *
 *     D(a || b) = a log(a/b) + (1-a) log((1-a)/(1-b))
 *
 * Both boundary cases are real here and both break the naive expression:
 * a community can contain every possible edge (a = 1) or none (a = 0), and
 * the corresponding term has a limit of 0 rather than the NaN that 0*log(0)
 * produces. Extracted so those limits are handled once and correctly rather
 * than twice and differently.
 */
final class KullbackLeibler
{
    public static function binary(float $observed, float $expected): float
    {
        // A degenerate reference distribution carries no information to
        // diverge from.
        if ($expected <= 0.0 || $expected >= 1.0) {
            return 0.0;
        }

        $divergence = 0.0;

        // x log(x/y) -> 0 as x -> 0, so an absent term contributes nothing.
        if ($observed > 0.0) {
            $divergence += $observed * log($observed / $expected);
        }

        if ($observed < 1.0) {
            $divergence += (1.0 - $observed) * log((1.0 - $observed) / (1.0 - $expected));
        }

        return $divergence;
    }
}
