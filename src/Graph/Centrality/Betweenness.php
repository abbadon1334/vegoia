<?php

declare(strict_types=1);

namespace Vegoia\Graph\Centrality;

use function abs;
use function array_fill;
use function count;

use SplPriorityQueue;
use Vegoia\Exception\InvalidArgument;
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
 * Two variants, answering different questions. `of()` counts hops: the
 * shortest path is the one crossing fewest edges. `weighted()` reads weights
 * as distances, so a heavy edge is a long detour and paths route around it --
 * the opposite of what a weight usually means elsewhere in this library, where
 * heavier is a stronger tie. Invert your weights before calling it if that is
 * what they represent.
 *
 * @see U. Brandes (2001), "A faster algorithm for betweenness centrality",
 *      Journal of Mathematical Sociology 25(2), 163-177.
 */
final class Betweenness
{
    /**
     * Relative slack when deciding whether two routes are equally short.
     *
     * Distances accumulated along different paths can be equal in exact
     * arithmetic and differ in their last bits. A strict comparison would then
     * count one of two equally short paths and drop the other, giving a
     * betweenness wrong by a whole path and entirely plausible-looking.
     */
    private const float TIE_TOLERANCE = 1.0e-12;

    /**
     * Betweenness by hop count.
     *
     * @return list<float>
     */
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

    /**
     * Betweenness with weights read as distances.
     *
     * The same dependency accumulation, but Dijkstra replaces the
     * breadth-first sweep, so the order nodes are finished in has to be
     * recorded explicitly: a node's dependency is complete only once every
     * node further from the source is done, and with weights "further" is no
     * longer "found later".
     *
     * @return list<float>
     */
    public static function weighted(Graph $graph): array
    {
        $order = $graph->order();

        if ($order === 0) {
            return [];
        }

        [$offsets, $targets, $weights] = $graph->csr();

        /** @var list<float> $score */
        $score = array_fill(0, $order, 0.0);

        for ($source = 0; $source < $order; $source++) {
            /** @var list<list<int>> $predecessors */
            $predecessors = array_fill(0, $order, []);
            /** @var list<float> $pathCount */
            $pathCount = array_fill(0, $order, 0.0);
            /** @var list<float> $distance */
            $distance = array_fill(0, $order, INF);
            $settled = array_fill(0, $order, false);

            $pathCount[$source] = 1.0;
            $distance[$source] = 0.0;

            $frontier = new SplPriorityQueue();
            $frontier->setExtractFlags(SplPriorityQueue::EXTR_DATA);
            $frontier->insert($source, 0.0);

            /** @var list<int> $finished */
            $finished = [];

            while (! $frontier->isEmpty()) {
                /** @var int $node */
                $node = $frontier->extract();

                if ($settled[$node]) {
                    continue;
                }

                $settled[$node] = true;
                $finished[] = $node;

                $end = $offsets[$node + 1];

                for ($i = $offsets[$node]; $i < $end; $i++) {
                    $weight = $weights[$i];

                    if ($weight < 0.0) {
                        throw InvalidArgument::malformedEdge(
                            'Weighted betweenness needs non-negative weights; edge '
                            . $node . '-' . $targets[$i] . ' has weight ' . $weight
                        );
                    }

                    $neighbour = $targets[$i];
                    $candidate = $distance[$node] + $weight;
                    $known = $distance[$neighbour];

                    if ($candidate < $known - self::TIE_TOLERANCE * $candidate) {
                        $distance[$neighbour] = $candidate;
                        $pathCount[$neighbour] = $pathCount[$node];
                        $predecessors[$neighbour] = [$node];
                        $frontier->insert($neighbour, -$candidate);
                    } elseif (abs($candidate - $known) <= self::TIE_TOLERANCE * $candidate) {
                        // Another shortest path of the same length.
                        $pathCount[$neighbour] += $pathCount[$node];
                        $predecessors[$neighbour][] = $node;
                    }
                }
            }

            /** @var list<float> $dependency */
            $dependency = array_fill(0, $order, 0.0);

            for ($i = count($finished) - 1; $i > 0; $i--) {
                $node = $finished[$i];
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

        /** @var list<float> $score */
        return $score;
    }
}
