<?php

declare(strict_types=1);

namespace Vegoia\Graph\Centrality;

use function array_fill;
use function count;

use Vegoia\Graph\Graph;

/**
 * Betweenness centrality by Brandes' algorithm.
 *
 * How often a node sits on a shortest path between two others. The naive route
 * is to enumerate all pairs of shortest paths, which is hopeless; Brandes
 * instead runs one breadth-first search per source and accumulates
 * dependencies backwards over the search order, bringing the cost down to
 * O(nm) with O(n + m) memory.
 *
 * Two conventions, both matching networkx's `normalized=False`:
 *
 *   * scores are raw pair counts, not divided by the number of pairs;
 *   * on an undirected graph each pair is discovered from both ends, so the
 *     totals are halved.
 *
 * Unweighted, by hop count. A weighted variant needs Dijkstra in place of the
 * breadth-first sweep, and answers a different question.
 *
 * @see U. Brandes (2001), "A faster algorithm for betweenness centrality",
 *      Journal of Mathematical Sociology 25(2), 163-177.
 */
final class Betweenness
{
    /** @return list<float> */
    public static function of(Graph $graph): array
    {
        $order = $graph->order();

        if ($order === 0) {
            return [];
        }

        [$offsets, $targets] = $graph->csr();

        /** @var list<float> $score */
        $score = array_fill(0, $order, 0.0);

        for ($source = 0; $source < $order; $source++) {
            /** @var list<list<int>> $predecessors */
            $predecessors = array_fill(0, $order, []);
            /** @var list<float> $pathCount */
            $pathCount = array_fill(0, $order, 0.0);
            /** @var list<int> $distance */
            $distance = array_fill(0, $order, -1);

            $pathCount[$source] = 1.0;
            $distance[$source] = 0;

            /** @var list<int> $order_of_discovery */
            $order_of_discovery = [];
            $queue = [$source];
            $head = 0;

            while ($head < count($queue)) {
                $node = $queue[$head++];
                $order_of_discovery[] = $node;
                $end = $offsets[$node + 1];

                for ($i = $offsets[$node]; $i < $end; $i++) {
                    $neighbour = $targets[$i];

                    if ($distance[$neighbour] < 0) {
                        $distance[$neighbour] = $distance[$node] + 1;
                        $queue[] = $neighbour;
                    }

                    // Every equally short route counts, which is why the
                    // dependency below is a share and not a whole.
                    if ($distance[$neighbour] === $distance[$node] + 1) {
                        $pathCount[$neighbour] += $pathCount[$node];
                        $predecessors[$neighbour][] = $node;
                    }
                }
            }

            /** @var list<float> $dependency */
            $dependency = array_fill(0, $order, 0.0);

            // Backwards over discovery order: a node's dependency is complete
            // only once every node further from the source has been handled.
            for ($i = count($order_of_discovery) - 1; $i > 0; $i--) {
                $node = $order_of_discovery[$i];
                $share = (1.0 + $dependency[$node]) / $pathCount[$node];

                foreach ($predecessors[$node] as $predecessor) {
                    $dependency[$predecessor] += $pathCount[$predecessor] * $share;
                }

                $score[$node] += $dependency[$node];
            }
        }

        if (! $graph->isDirected()) {
            for ($node = 0; $node < $order; $node++) {
                $score[$node] /= 2.0;
            }
        }

        /** @var list<float> $score every index 0..order-1 was filled above */
        return $score;
    }
}
