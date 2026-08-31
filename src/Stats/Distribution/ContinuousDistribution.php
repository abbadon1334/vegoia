<?php

declare(strict_types=1);

namespace Vegoia\Stats\Distribution;

use function abs;
use function is_finite;
use function log;

use Vegoia\Exception\InvalidArgument;

/**
 * Inverts a distribution from its own density and survival function.
 *
 * Each distribution supplies a starting point it knows how to guess well;
 * everything below is shared, because the hard part is not the guess but
 * staying accurate in the tail the guess is worst in.
 *
 * The iteration is Newton's method on log(survival(x)) - log(p) rather than on
 * survival(x) - p. In the far tail the survival function falls by orders of
 * magnitude across a narrow range of x, so the untransformed residual is a
 * difference of two numbers that are both nearly zero and carries almost no
 * information about how far away the root is; its logarithm is close to linear
 * there, and the same step lands. Working from the survival function rather
 * than the cumulative is the same decision one level down: at p = 1e-300 the
 * cumulative is 1 and has nothing left to say.
 *
 * Newton alone is not enough, because these functions are not convex
 * everywhere and a step can leave the region entirely. Every step is therefore
 * kept inside a bracket that is known to contain the root, and a step that
 * escapes is replaced by a bisection. That is slower to write and cannot fail
 * to converge.
 */
abstract class ContinuousDistribution implements Distribution
{
    /** Newton converges quadratically; this is a ceiling, not an expectation. */
    private const int MAX_ITERATIONS = 200;

    /** Relative width at which the bracket cannot be narrowed further. */
    private const float TOLERANCE = 1.0e-15;

    /**
     * A first approximation to the x with P(X > x) = p.
     *
     * It only has to be in the right region; the bracket search widens it and
     * Newton refines it. A good guess saves iterations, never correctness.
     */
    abstract protected function guessUpperQuantile(float $p): float;

    /** The infimum of the support: what P(X > x) = 1 means as an x. */
    abstract protected function infimum(): float;

    /**
     * The point the bracket search expands away from.
     *
     * Zero for all four distributions here: two are supported on the positive
     * reals, and the two symmetric ones reduce a probability above a half to
     * its reflection before searching, so the search only ever runs on the
     * non-negative half. It is separate from the infimum because that one is
     * -INF for those two, which no doubling can start from.
     */
    protected function searchOrigin(): float
    {
        return 0.0;
    }

    public function quantile(float $p): float
    {
        self::assertProbability($p);

        // Asked against the tail that has the digits: for p above a half the
        // complement is the small number and is the one worth computing.
        return $this->upperQuantile(1.0 - $p);
    }

    public function upperQuantile(float $p): float
    {
        self::assertProbability($p);

        if ($p === 0.0) {
            return INF;
        }

        if ($p === 1.0) {
            return $this->infimum();
        }

        $x = $this->guessUpperQuantile($p);

        if (! is_finite($x)) {
            $x = $this->searchOrigin() + 1.0;
        }

        [$low, $high] = $this->bracket($x, $p);
        $x = min(max($x, $low), $high);

        $target = log($p);

        for ($i = 0; $i < self::MAX_ITERATIONS; $i++) {
            $survival = $this->survival($x);
            $density = $this->density($x);

            // Keep the bracket honest before stepping: survival decreases, so
            // a point with too much mass above it is a new lower bound.
            if ($survival > $p) {
                $low = $x;
            } else {
                $high = $x;
            }

            if ($high - $low <= self::TOLERANCE * max(abs($low), abs($high), 1.0)) {
                break;
            }

            $next = $density > 0.0 && $survival > 0.0
                ? $x + (log($survival) - $target) * $survival / $density
                : INF;

            // Outside the bracket, or not a number at all: bisect instead.
            // Bisection halves the interval every time and cannot diverge,
            // which is what makes the loop terminate rather than hope to.
            if (! is_finite($next) || $next <= $low || $next >= $high) {
                $next = $low + ($high - $low) / 2.0;
            }

            if ($next === $x) {
                break;
            }

            $x = $next;
        }

        return $x;
    }

    /**
     * Widen the guess into an interval that contains the root.
     *
     * Doubling the distance from the support, rather than adding a fixed
     * step, because the distance from the support to the 1e-300 quantile
     * spans eleven orders of magnitude between Student's t with one degree of
     * freedom and with a thousand.
     *
     * @return array{float, float}
     */
    private function bracket(float $guess, float $p): array
    {
        $origin = $this->searchOrigin();
        $step = max(abs($guess - $origin), 1.0);

        $low = $origin;
        $high = $guess;

        // 2000 rather than a smaller ceiling because Student's t with one
        // degree of freedom puts its 1e-12 quantile at 3e11 while the normal
        // puts it at 7, and the same doubling has to reach both.
        for ($i = 0; $i < 2000 && $this->survival($high) > $p; $i++) {
            $low = $high;
            $high = $origin + ($high - $origin) * 2.0 + $step;
        }

        // The guess overshot: walk back towards the origin instead.
        if ($low === $origin) {
            $candidate = $guess;

            for ($i = 0; $i < 2000 && $this->survival($candidate) < $p; $i++) {
                $high = $candidate;
                $candidate = $origin + ($candidate - $origin) / 2.0;

                if ($candidate === $origin) {
                    break;
                }
            }

            $low = $candidate;
        }

        return [$low, $high];
    }

    protected static function assertProbability(float $p): void
    {
        if (! ($p >= 0.0 && $p <= 1.0)) {
            throw InvalidArgument::outOfRange('a probability', $p, 0.0, 1.0);
        }
    }

    protected static function assertPositiveShape(string $what, float $value): void
    {
        if ($value <= 0.0 || ! is_finite($value)) {
            throw InvalidArgument::outOfDomain($what, $value, 'it must be positive and finite');
        }
    }
}
