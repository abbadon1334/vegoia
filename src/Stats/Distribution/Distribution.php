<?php

declare(strict_types=1);

namespace Vegoia\Stats\Distribution;

/**
 * A continuous distribution, with both tails treated as first-class.
 *
 * The pairing of cumulative with survival, and quantile with upperQuantile,
 * is the whole reason this interface has five methods where three would look
 * sufficient. A p-value is a tail probability, and past about four standard
 * deviations the upper tail cannot be recovered from the lower one: at z = 10
 * the cumulative is 1 to every bit a double has, so `1 - cumulative(10)` is
 * exactly zero where the true survival is 7.6e-24. Reporting p = 0 there is
 * not a rounding error, it is a different claim.
 *
 * So implementations compute each tail by its own route, and the inverse is
 * offered against the tail it inverts. `upperQuantile(1e-300)` is an ordinary
 * question with the answer 37.0; `quantile(1 - 1e-300)` cannot even be asked,
 * because the argument is 1.
 */
interface Distribution
{
    /** The probability density at x. */
    public function density(float $x): float;

    /** P(X <= x). */
    public function cumulative(float $x): float;

    /** P(X > x), computed directly rather than as 1 - cumulative(x). */
    public function survival(float $x): float;

    /** The x with P(X <= x) = p. */
    public function quantile(float $p): float;

    /** The x with P(X > x) = p, which is where a p-value asks its question. */
    public function upperQuantile(float $p): float;
}
