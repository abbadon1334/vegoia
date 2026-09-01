<?php

declare(strict_types=1);

namespace Vegoia\Graph;

use Vegoia\Exception\InvalidArgument;
use Vegoia\Stats\Correlation;

/**
 * Do nodes attach to their own kind?
 *
 * Newman's degree assortativity is Pearson's correlation between the degrees
 * at the two ends of an edge, over every edge. Positive means hubs join hubs
 * and leaves join leaves, which is what social networks look like; negative
 * means hubs join leaves, which is what most technological and biological ones
 * look like. Zachary's karate club is -0.4756: the two instructors are joined
 * to students, not to each other.
 */
final class Assortativity
{
    /**
     * @throws InvalidArgument on a regular graph, where it is undefined
     */
    public static function degree(Graph $graph): float
    {
        if ($graph->isDirected()) {
            throw InvalidArgument::directedNotSupported(
                'Degree assortativity',
                'a directed graph has four of these -- in-in, in-out, out-in and out-out -- and '
                . 'they are different measures rather than one measure computed four ways.',
            );
        }

        $left = [];
        $right = [];

        foreach ($graph->edges() as [$from, $to]) {
            // A self-loop says a node agrees with itself, which is true of
            // every node and so carries no information about what anything
            // prefers to attach to. Including it would force a (d, d) pair and
            // drag the correlation mechanically towards +1.
            if ($from === $to) {
                continue;
            }

            // Both orientations, because an undirected edge has no first end.
            // Symmetrising is what makes this Newman's coefficient rather than
            // a correlation that depends on how the edges happened to be
            // stored.
            $left[] = (float) $graph->degree($from);
            $right[] = (float) $graph->degree($to);
            $left[] = (float) $graph->degree($to);
            $right[] = (float) $graph->degree($from);
        }

        if ($left === []) {
            throw InvalidArgument::malformedEdge(
                'Degree assortativity needs at least one edge that is not a self-loop; there is '
                . 'nothing to correlate'
            );
        }

        // Correlation::pearson brings the deviation form, compensated
        // accumulation, and the clamp to [-1, 1] whose comment already says
        // rounding can carry a perfect correlation a hair past 1 -- which
        // happens here, on three disjoint cliques where every edge joins equal
        // degrees and the true answer is exactly 1.
        //
        // It also brings the guard: a regular graph has no variation in
        // endpoint degree, so the correlation has no denominator, and pearson
        // already refuses that with the right words.
        return Correlation::pearson($left, $right);
    }
}
