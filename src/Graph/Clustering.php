<?php

declare(strict_types=1);

namespace Vegoia\Graph;

use function array_fill;
use function intdiv;

use Vegoia\Support\CompensatedSum;

/**
 * Triangles, and the two ways of averaging them.
 *
 * The local clustering coefficient of a node is the fraction of its
 * neighbour pairs that are themselves joined -- how close its neighbourhood
 * comes to being a clique. It answers "are my friends friends with each
 * other", which is the question that separates a real social network from a
 * random graph of the same density.
 *
 * The two summaries are different measures and are routinely confused:
 *
 *   * averageCoefficient() is the mean of the per-node coefficients, so every
 *     node counts equally and low-degree nodes -- where the coefficient is
 *     computed over one or three pairs -- dominate the average.
 *   * transitivity() is 3 * triangles / connected-triples, a single global
 *     ratio, so high-degree nodes contribute in proportion to how many triples
 *     they sit in.
 *
 * On a graph with many low-degree nodes they can differ by a factor of two.
 * Both are reported here so the choice is explicit.
 *
 * Purely topological: weights are ignored, as in the standard definitions.
 */
final class Clustering
{
    /**
     * Triangles through each node.
     *
     * Counted by walking the neighbour lists, which the CSR layout keeps
     * sorted, so the shared neighbours of an edge's endpoints are found by a
     * linear merge rather than a hash lookup per candidate.
     *
     * @return list<int>
     */
    public static function triangles(Graph $graph): array
    {
        $order = $graph->order();

        if ($order === 0) {
            return [];
        }

        [$offsets, $targets] = $graph->csr();

        /** @var list<int> $count */
        $count = array_fill(0, $order, 0);

        for ($node = 0; $node < $order; $node++) {
            $end = $offsets[$node + 1];

            for ($i = $offsets[$node]; $i < $end; $i++) {
                $neighbour = $targets[$i];

                // Each triangle would otherwise be found from all three
                // corners and in both directions.
                if ($neighbour <= $node) {
                    continue;
                }

                $shared = self::sharedNeighbours($offsets, $targets, $node, $neighbour);

                foreach ($shared as $third) {
                    if ($third > $neighbour) {
                        $count[$node]++;
                        $count[$neighbour]++;
                        $count[$third]++;
                    }
                }
            }
        }

        return $count;
    }

    /** Distinct triangles in the whole graph. */
    public static function triangleCount(Graph $graph): int
    {
        $total = 0;

        foreach (self::triangles($graph) as $count) {
            $total += $count;
        }

        // Each triangle was counted once at each of its three corners.
        return intdiv($total, 3);
    }

    /**
     * Local clustering coefficient per node. A node of degree below 2 has no
     * neighbour pairs and scores 0 -- not undefined, by the usual convention.
     *
     * @return list<float>
     */
    public static function coefficients(Graph $graph): array
    {
        $order = $graph->order();
        $triangles = self::triangles($graph);

        /** @var list<float> $coefficient */
        $coefficient = array_fill(0, $order, 0.0);

        for ($node = 0; $node < $order; $node++) {
            $degree = self::simpleDegree($graph, $node);

            if ($degree < 2) {
                continue;
            }

            $coefficient[$node] = 2.0 * $triangles[$node] / ($degree * ($degree - 1));
        }

        return $coefficient;
    }

    /** The mean of the local coefficients: every node counts equally. */
    public static function averageCoefficient(Graph $graph): float
    {
        $order = $graph->order();

        if ($order === 0) {
            return 0.0;
        }

        $sum = new CompensatedSum();

        foreach (self::coefficients($graph) as $coefficient) {
            $sum->add($coefficient);
        }

        return $sum->value() / $order;
    }

    /**
     * Global transitivity: 3 * triangles / connected triples. Weighted towards
     * high-degree nodes, unlike averageCoefficient().
     */
    public static function transitivity(Graph $graph): float
    {
        $triples = 0;

        for ($node = 0; $node < $graph->order(); $node++) {
            $degree = self::simpleDegree($graph, $node);
            $triples += $degree * ($degree - 1) / 2;
        }

        if ($triples === 0) {
            return 0.0;
        }

        return 3.0 * self::triangleCount($graph) / $triples;
    }

    /**
     * Distinct neighbours, self-loop excluded: a node is not its own
     * neighbour for the purpose of counting triangles.
     */
    private static function simpleDegree(Graph $graph, int $node): int
    {
        $degree = 0;

        foreach ($graph->neighbours($node) as $neighbour => $_) {
            if ($neighbour !== $node) {
                $degree++;
            }
        }

        return $degree;
    }

    /**
     * @param  list<int> $offsets
     * @param  list<int> $targets
     * @return list<int> neighbours shared by both nodes, ascending
     */
    private static function sharedNeighbours(array $offsets, array $targets, int $left, int $right): array
    {
        $i = $offsets[$left];
        $endLeft = $offsets[$left + 1];
        $j = $offsets[$right];
        $endRight = $offsets[$right + 1];

        $shared = [];

        while ($i < $endLeft && $j < $endRight) {
            $a = $targets[$i];
            $b = $targets[$j];

            if ($a === $b) {
                if ($a !== $left && $a !== $right) {
                    $shared[] = $a;
                }
                $i++;
                $j++;
            } elseif ($a < $b) {
                $i++;
            } else {
                $j++;
            }
        }

        return $shared;
    }
}
