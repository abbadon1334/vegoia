<?php

declare(strict_types=1);

namespace Vegoia\Graph;

use function array_fill;
use function array_pop;
use function count;

/**
 * Connectivity questions answered with an iterative breadth-first sweep.
 *
 * Iterative rather than recursive on purpose: a long path in a large graph
 * would otherwise be limited by the call stack, and PHP gives no useful
 * diagnostic when it runs out.
 *
 * `inducesConnectedSubgraph()` is the reason this class exists at all. The
 * guarantee that distinguishes Leiden from Louvain is that every community it
 * returns is internally connected -- Louvain can and does return communities
 * split into pieces that are only joined through nodes outside them. That
 * property is checkable directly, with no golden values involved, so it is the
 * strongest assertion available about a stochastic algorithm.
 */
final class Connectivity
{
    public static function isConnected(Graph $graph): bool
    {
        return $graph->order() === 0 || self::components($graph)->count() === 1;
    }

    public static function components(Graph $graph): Partition
    {
        $order = $graph->order();

        if ($order === 0) {
            return Partition::fromMembership([]);
        }

        [$offsets, $targets] = $graph->csr();

        /** @var list<int> $component */
        $component = array_fill(0, $order, -1);
        $found = 0;

        for ($root = 0; $root < $order; $root++) {
            if ($component[$root] !== -1) {
                continue;
            }

            $component[$root] = $found;
            $frontier = [$root];

            while ($frontier !== []) {
                $node = array_pop($frontier);
                $end = $offsets[$node + 1];

                for ($i = $offsets[$node]; $i < $end; $i++) {
                    $neighbour = $targets[$i];

                    if ($component[$neighbour] === -1) {
                        $component[$neighbour] = $found;
                        $frontier[] = $neighbour;
                    }
                }
            }

            $found++;
        }

        return Partition::fromMembership($component);
    }

    /**
     * Is the subgraph induced by these nodes connected?
     *
     * Only edges with *both* endpoints inside the set count -- a path leaving
     * the set and coming back does not make it connected, which is exactly the
     * distinction Leiden's guarantee turns on.
     *
     * @param list<int> $nodes
     */
    public static function inducesConnectedSubgraph(Graph $graph, array $nodes): bool
    {
        $size = count($nodes);

        if ($size <= 1) {
            return true;
        }

        $inside = [];
        foreach ($nodes as $node) {
            $inside[$node] = true;
        }

        [$offsets, $targets] = $graph->csr();

        $seen = [$nodes[0] => true];
        $frontier = [$nodes[0]];
        $reached = 1;

        while ($frontier !== []) {
            $node = array_pop($frontier);
            $end = $offsets[$node + 1];

            for ($i = $offsets[$node]; $i < $end; $i++) {
                $neighbour = $targets[$i];

                if (! isset($inside[$neighbour]) || isset($seen[$neighbour])) {
                    continue;
                }

                $seen[$neighbour] = true;
                $frontier[] = $neighbour;
                $reached++;
            }
        }

        return $reached === $size;
    }
}
