<?php

declare(strict_types=1);

namespace Vegoia\Graph\Centrality;

use function array_fill;

use Vegoia\Graph\Graph;
use Vegoia\Graph\Path\BreadthFirst;

/**
 * Closeness centrality with the Wasserman-Faust correction.
 *
 * The textbook definition -- the reciprocal of the mean distance to everyone
 * else -- silently assumes the graph is connected. On a disconnected graph the
 * distance to an unreachable node is infinite, so either the score collapses to
 * zero for every node, or the unreachable ones are quietly dropped, which
 * rewards nodes in small components for having little to be far from.
 *
 * The correction scales by the fraction of the graph a node can actually
 * reach:
 *
 *     C(u) = (r / sum of distances) * (r / (n - 1)),  r = nodes reachable from u
 *
 * so reaching a few nodes quickly no longer outranks reaching most of the graph
 * at reasonable cost. This is networkx's `wf_improved=True`, its default.
 *
 * @see S. Wasserman & K. Faust (1994), Social Network Analysis, 110-111.
 */
final class Closeness
{
    /** @return list<float> */
    public static function of(Graph $graph): array
    {
        $order = $graph->order();

        if ($order === 0) {
            return [];
        }

        /** @var list<float> $score */
        $score = array_fill(0, $order, 0.0);

        for ($node = 0; $node < $order; $node++) {
            $distances = BreadthFirst::distancesFrom($graph, $node);

            $total = 0.0;
            $reachable = 0;

            foreach ($distances as $other => $distance) {
                if ($other === $node || $distance < 0.0) {
                    continue;
                }

                $total += $distance;
                $reachable++;
            }

            if ($reachable === 0 || $total === 0.0 || $order === 1) {
                continue;
            }

            $score[$node] = ($reachable / $total) * ($reachable / ($order - 1));
        }

        /** @var list<float> $score every index 0..order-1 was filled above */
        return $score;
    }
}
