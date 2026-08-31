<?php

declare(strict_types=1);

namespace Vegoia\Graph;

use function array_fill;
use function usort;

use Vegoia\Exception\InvalidArgument;

/**
 * The cheapest, or dearest, set of edges that still joins everything.
 *
 * The textbook use is the cheap one: lay cable to every house for as little
 * as possible. The use here is usually the other way round. Build a similarity
 * graph -- k nearest neighbours over embeddings, cosine over TF-IDF, whatever
 * -- and you get something dense and noisy where almost every pair has some
 * small resemblance. The *maximum* spanning tree keeps the strongest link out
 * of every node and discards the rest, which turns that into a skeleton small
 * enough to look at and sparse enough to run a layout on.
 *
 * Kruskal, with union-find. Sort the edges once, then walk them in order
 * taking any edge that joins two pieces that were not already joined. The
 * union-find is what makes "were they already joined" cost almost nothing:
 * with path compression and union by size, the amortised cost of a query is
 * the inverse Ackermann function, which is below 5 for any graph that will
 * ever exist.
 *
 * A disconnected graph gives a forest, one tree per component, and that is the
 * general case rather than an error to refuse: a knowledge graph with two
 * unrelated clusters has no spanning tree and its spanning forest is still the
 * right answer. The result has n - c edges, not n - 1.
 *
 * The tree is not unique when weights tie, and on an unweighted graph every
 * weight ties. Two correct implementations will disagree about which edges
 * they picked and agree exactly about the total weight, which is why the tests
 * assert the latter.
 */
final class SpanningTree
{
    /**
     * The edges of a minimum spanning forest, each with the lower endpoint
     * first.
     *
     * @return list<array{int, int, float}>
     */
    public static function minimum(Graph $graph): array
    {
        return self::kruskal($graph, dearest: false);
    }

    /**
     * The edges of a maximum spanning forest.
     *
     * The same algorithm with the sort reversed, which is correct rather than
     * a trick: Kruskal's exchange argument never assumes the order is
     * ascending, only that it is consistent.
     *
     * @return list<array{int, int, float}>
     */
    public static function maximum(Graph $graph): array
    {
        return self::kruskal($graph, dearest: true);
    }

    /**
     * The same forest, as a graph you can carry on computing with.
     *
     * @param list<array{int, int, float}> $edges from minimum() or maximum()
     */
    public static function asGraph(Graph $graph, array $edges): Graph
    {
        return Graph::undirected($graph->order(), $edges);
    }

    /** What the forest costs, which is the part two implementations agree on. */
    public static function weight(Graph $graph, bool $dearest = false): float
    {
        $total = 0.0;

        foreach (self::kruskal($graph, $dearest) as [, , $weight]) {
            $total += $weight;
        }

        return $total;
    }

    /**
     * @return list<array{int, int, float}>
     */
    private static function kruskal(Graph $graph, bool $dearest): array
    {
        if ($graph->isDirected()) {
            throw InvalidArgument::directedNotSupported(
                'A spanning tree',
                'the directed analogue is a minimum spanning arborescence, rooted at a chosen '
                . 'node and found by a different algorithm entirely.',
            );
        }

        $order = $graph->order();

        if ($order === 0) {
            return [];
        }

        $edges = [];

        // Self-loops can never join two pieces, so they would be rejected
        // below anyway; skipping them here keeps the sort smaller.
        foreach ($graph->edges() as [$from, $to, $weight]) {
            if ($from !== $to) {
                $edges[] = [$from, $to, $weight];
            }
        }

        // Ties broken by endpoint, so the answer does not depend on the order
        // the edges happened to be stored in. It stays one of several correct
        // trees; it is simply always the same one.
        usort($edges, static function (array $a, array $b) use ($dearest): int {
            $byWeight = $dearest ? $b[2] <=> $a[2] : $a[2] <=> $b[2];

            return $byWeight !== 0 ? $byWeight : ($a[0] <=> $b[0] ?: $a[1] <=> $b[1]);
        });

        $parent = [];
        $size = array_fill(0, $order, 1);

        for ($node = 0; $node < $order; $node++) {
            $parent[$node] = $node;
        }

        $forest = [];
        $joined = 0;

        foreach ($edges as [$from, $to, $weight]) {
            $left = self::find($parent, $from);
            $right = self::find($parent, $to);

            if ($left === $right) {
                continue;
            }

            // Union by size: hang the smaller tree under the larger, which is
            // what keeps the structure shallow and find() cheap.
            if ($size[$left] < $size[$right]) {
                [$left, $right] = [$right, $left];
            }

            $parent[$right] = $left;
            $size[$left] += $size[$right];

            $forest[] = [$from, $to, $weight];

            // A forest of n nodes over c components has n - c edges, and once
            // it does no later edge can be taken.
            if (++$joined === $order - 1) {
                break;
            }
        }

        return $forest;
    }

    /**
     * The representative of a node's piece, compressing the path on the way.
     *
     * Iterative: the recursive form is one frame per level, and while union by
     * size keeps that shallow, "shallow" is not "bounded".
     *
     * @param array<int, int> $parent
     */
    private static function find(array &$parent, int $node): int
    {
        $root = $node;

        while ($parent[$root] !== $root) {
            $root = $parent[$root];
        }

        while ($parent[$node] !== $root) {
            [$node, $parent[$node]] = [$parent[$node], $root];
        }

        return $root;
    }
}
