<?php

declare(strict_types=1);

namespace Vegoia\Exception;

use RuntimeException;

use function sprintf;

/**
 * An iterative method ran out of iterations before it settled.
 *
 * Raised rather than returned, because the alternative is worse. A power
 * iteration that has not converged still holds a vector of plausible-looking
 * numbers that sum to one, and nothing about it says it is halfway to
 * somewhere else. This library already had that bug once: HITS was measuring
 * the drift of only one of its two coupled vectors, declared victory after a
 * single iteration on a graph where the other had not moved yet, and returned
 * [0.2, 0.2, 0.4, 0.2, 0] where the answer is [0, 0, 1, 0, 0].
 *
 * Katz has always refused rather than guessed, for its own divergence case.
 * The rest now agree with it: same class of failure, same answer.
 *
 * On every graph in the test collection -- up to a thousand nodes and sixteen
 * thousand edges -- the defaults converge in a small fraction of the iterations
 * allowed. Reaching the ceiling therefore means something unusual about the
 * graph rather than something ordinary about its size, which is why this says
 * what it reached and what it needed rather than only that it gave up.
 */
final class DidNotConverge extends RuntimeException implements VegoiaException
{
    /**
     * @param string $method     what was being computed
     * @param int    $iterations the ceiling that was reached
     * @param float  $reached    the drift the last step still had
     * @param float  $required   the drift it had to get below
     */
    public static function after(string $method, int $iterations, float $reached, float $required): self
    {
        return new self(sprintf(
            '%s did not converge in %d iterations: the last step still moved by %.3g and had to '
            . 'move by less than %.3g. Either raise the iteration ceiling, or loosen the '
            . 'tolerance if that much accuracy is not needed. On an ordinary graph this does not '
            . 'happen, so a graph that provokes it is worth looking at.',
            $method,
            $iterations,
            $reached,
            $required,
        ));
    }
}
