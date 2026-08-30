<?php

declare(strict_types=1);

namespace Vegoia\Graph\Path;

use function array_fill;
use function count;

use Vegoia\Exception\InvalidArgument;
use Vegoia\Graph\Graph;

/**
 * Shortest paths counted in hops, ignoring weights.
 *
 * Unreachable nodes come back as -1 rather than INF or null: the result is a
 * packed list of floats that callers can index without unwrapping, and -1 is
 * unmistakable next to a genuine distance.
 */
final class BreadthFirst
{
    /** @return list<float> distance in hops, -1 where unreachable */
    public static function distancesFrom(Graph $graph, int $source): array
    {
        $order = $graph->order();

        if ($source < 0 || $source >= $order) {
            throw InvalidArgument::nodeOutOfRange($source, $order);
        }

        [$offsets, $targets] = $graph->csr();

        /** @var list<float> $distance */
        $distance = array_fill(0, $order, -1.0);
        $distance[$source] = 0.0;

        $queue = [$source];
        $head = 0;

        while ($head < count($queue)) {
            $node = $queue[$head++];
            $next = $distance[$node] + 1.0;
            $end = $offsets[$node + 1];

            for ($i = $offsets[$node]; $i < $end; $i++) {
                $neighbour = $targets[$i];

                if ($distance[$neighbour] < 0.0) {
                    $distance[$neighbour] = $next;
                    $queue[] = $neighbour;
                }
            }
        }

        /** @var list<float> $distance array_fill over 0..order-1 */
        return $distance;
    }
}
