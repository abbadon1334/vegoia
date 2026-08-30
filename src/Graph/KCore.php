<?php

declare(strict_types=1);

namespace Vegoia\Graph;

use function array_fill;
use function array_key_first;
use function max;

/**
 * k-core decomposition: how deep in the graph each node sits.
 *
 * The k-core is the largest subgraph in which every node has at least k
 * neighbours *within it*. A node's core number is the largest k whose core
 * contains it, and that single integer says something degree cannot: a node
 * with a hundred edges to leaves has core number 1, while a node with four
 * edges into a dense cluster has four. It measures embedding rather than
 * popularity.
 *
 * Useful for pruning before expensive work. Community detection and
 * betweenness on a large graph often spend most of their time on the
 * periphery, and dropping the 1-core is a cheap, principled way to remove it
 * without arbitrary thresholds.
 *
 * Computed by repeated peeling: strip the lowest-degree node, record the
 * running maximum of the degrees seen, decrement its neighbours, repeat. The
 * bucket queue makes it linear in the number of edges -- sorting after every
 * removal would make it O(m log n) for no benefit.
 *
 * @see V. Batagelj & M. Zaversnik (2003), "An O(m) Algorithm for Cores
 *      Decomposition of Networks", arXiv:cs/0310049.
 */
final class KCore
{
    /**
     * Core number per node.
     *
     * Topological: a self-loop does not make a node better connected to the
     * rest of the graph, so it is not counted.
     *
     * @return list<int>
     */
    public static function coreNumbers(Graph $graph): array
    {
        $order = $graph->order();

        if ($order === 0) {
            return [];
        }

        [$offsets, $targets] = $graph->csr();

        $degree = array_fill(0, $order, 0);
        $maximum = 0;

        for ($node = 0; $node < $order; $node++) {
            $end = $offsets[$node + 1];
            $count = 0;

            for ($i = $offsets[$node]; $i < $end; $i++) {
                if ($targets[$i] !== $node) {
                    $count++;
                }
            }

            $degree[$node] = $count;
            $maximum = max($maximum, $count);
        }

        // Bucket queue: buckets[d] holds the nodes currently of degree d, so
        // finding the minimum is a scan over buckets rather than over nodes.
        $buckets = array_fill(0, $maximum + 1, []);

        for ($node = 0; $node < $order; $node++) {
            $buckets[$degree[$node]][$node] = true;
        }

        /** @var list<int> $core */
        $core = array_fill(0, $order, 0);
        $removed = array_fill(0, $order, false);
        $level = 0;

        for ($processed = 0; $processed < $order; $processed++) {
            // The lowest non-empty bucket. The scan never moves backwards past
            // a node's own level, which is what keeps the whole loop linear.
            while ($level <= $maximum && $buckets[$level] === []) {
                $level++;
            }

            if ($level > $maximum) {
                break;
            }

            $node = array_key_first($buckets[$level]);
            unset($buckets[$level][$node]);
            $removed[$node] = true;
            $core[$node] = $level;

            $end = $offsets[$node + 1];

            for ($i = $offsets[$node]; $i < $end; $i++) {
                $neighbour = $targets[$i];

                if ($neighbour === $node || $removed[$neighbour]) {
                    continue;
                }

                // Losing an edge can only lower a neighbour's degree, never
                // below the current level -- so the bucket scan stays monotone.
                unset($buckets[$degree[$neighbour]][$neighbour]);
                $degree[$neighbour] = max($level, $degree[$neighbour] - 1);
                $buckets[$degree[$neighbour]][$neighbour] = true;
            }
        }

        return $core;
    }

    /** The largest k for which a non-empty k-core exists. */
    public static function degeneracy(Graph $graph): int
    {
        $maximum = 0;

        foreach (self::coreNumbers($graph) as $core) {
            $maximum = max($maximum, $core);
        }

        return $maximum;
    }

    /**
     * The nodes of the k-core, ascending. Useful for pruning a graph before
     * an expensive computation.
     *
     * @return list<int>
     */
    public static function nodesInCore(Graph $graph, int $k): array
    {
        $nodes = [];

        foreach (self::coreNumbers($graph) as $node => $core) {
            if ($core >= $k) {
                $nodes[] = $node;
            }
        }

        return $nodes;
    }
}
