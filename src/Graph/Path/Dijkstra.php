<?php

declare(strict_types=1);

namespace Vegoia\Graph\Path;

use function array_fill;
use function array_reverse;

use SplPriorityQueue;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Graph\Graph;

/**
 * Shortest paths by weight.
 *
 * Negative weights are rejected outright. Dijkstra's correctness rests on the
 * assumption that extending a path cannot shorten it, and with a negative edge
 * it will return a plausible wrong answer rather than fail -- the worst
 * possible behaviour for a number that ends up in a report. If you need
 * negative weights you need Bellman-Ford, which is a different algorithm and
 * not a flag on this one.
 *
 * Settled nodes are skipped by comparing against the recorded distance rather
 * than by decrease-key, which SplPriorityQueue does not offer: stale entries
 * are left in the heap and discarded on arrival.
 */
final class Dijkstra
{
    /** @return list<float> distance by weight, -1 where unreachable */
    public static function distancesFrom(Graph $graph, int $source): array
    {
        return self::run($graph, $source)[0];
    }

    /**
     * The nodes along a cheapest path, source first, or an empty list when
     * none exists.
     *
     * @return list<int>
     */
    public static function shortestPath(Graph $graph, int $source, int $destination): array
    {
        if ($destination < 0 || $destination >= $graph->order()) {
            throw InvalidArgument::nodeOutOfRange($destination, $graph->order());
        }

        [$distance, $previous] = self::run($graph, $source);

        if ($distance[$destination] < 0.0) {
            return [];
        }

        $path = [$destination];

        for ($node = $destination; $node !== $source; $node = $previous[$node]) {
            $path[] = $previous[$node];
        }

        return array_reverse($path);
    }

    /** @return array{list<float>, list<int>} */
    private static function run(Graph $graph, int $source): array
    {
        $order = $graph->order();

        if ($source < 0 || $source >= $order) {
            throw InvalidArgument::nodeOutOfRange($source, $order);
        }

        [$offsets, $targets, $weights] = $graph->csr();

        /** @var list<float> $distance */
        $distance = array_fill(0, $order, -1.0);
        /** @var list<int> $previous */
        $previous = array_fill(0, $order, -1);
        $settled = array_fill(0, $order, false);

        $distance[$source] = 0.0;

        $frontier = new SplPriorityQueue();
        $frontier->setExtractFlags(SplPriorityQueue::EXTR_DATA);
        // The queue is a max-heap, so priorities are negated distances.
        $frontier->insert($source, 0.0);

        while (! $frontier->isEmpty()) {
            /** @var int $node */
            $node = $frontier->extract();

            if ($settled[$node]) {
                continue;
            }

            $settled[$node] = true;
            $end = $offsets[$node + 1];

            for ($i = $offsets[$node]; $i < $end; $i++) {
                $weight = $weights[$i];

                if ($weight < 0.0) {
                    throw InvalidArgument::malformedEdge(
                        'Dijkstra needs non-negative weights; edge '
                        . $node . '-' . $targets[$i] . ' has weight ' . $weight
                    );
                }

                $neighbour = $targets[$i];
                $candidate = $distance[$node] + $weight;

                if ($distance[$neighbour] < 0.0 || $candidate < $distance[$neighbour]) {
                    $distance[$neighbour] = $candidate;
                    $previous[$neighbour] = $node;
                    $frontier->insert($neighbour, -$candidate);
                }
            }
        }

        /**
         * @var list<float> $distance
         * @var list<int>   $previous
         */
        return [$distance, $previous];
    }
}
