<?php

declare(strict_types=1);

namespace Vegoia\Graph\Centrality;

use function array_fill;

use Vegoia\Graph\Graph;
use Vegoia\Graph\Path\BreadthFirst;
use Vegoia\Support\CompensatedSum;

/**
 * Harmonic centrality: the sum of reciprocal distances.
 *
 *     H(u) = sum over v != u of 1 / d(u, v)
 *
 * The repair for closeness on a disconnected graph. Closeness sums distances
 * and so is undefined the moment one is infinite; here an unreachable node
 * contributes 1/inf = 0 and the sum simply carries on. No correction factor,
 * no special case -- which is why it is often the better choice on real
 * networks, where something is almost always unreachable from something.
 *
 * Unnormalised, matching networkx: divide by n-1 for a value in [0, 1].
 *
 * @see M. Marchiori & V. Latora (2000), "Harmony in the small-world",
 *      Physica A 285, 539-546.
 */
final class Harmonic
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
            $sum = new CompensatedSum();

            foreach (BreadthFirst::distancesFrom($graph, $node) as $other => $distance) {
                // Unreachable is -1, and contributes nothing rather than
                // poisoning the whole sum the way closeness would.
                if ($other !== $node && $distance > 0.0) {
                    $sum->add(1.0 / $distance);
                }
            }

            $score[$node] = $sum->value();
        }

        return $score;
    }
}
