<?php

declare(strict_types=1);

namespace Vegoia\Graph\Centrality;

use function abs;
use function array_fill;

use Vegoia\Exception\DidNotConverge;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Graph\Graph;

/**
 * PageRank by power iteration.
 *
 * The score is the stationary distribution of a random walk that, with
 * probability `1 - damping`, teleports to a uniformly chosen node instead of
 * following an edge. Teleportation is not a detail: without it the walk gets
 * trapped in whatever part of the graph has no way out, and on a disconnected
 * graph there is no unique answer at all.
 *
 * Conventions, matching networkx so the fixtures mean something:
 *
 *   * edge weights are used, normalised by each node's outgoing weight;
 *   * a node with no outgoing weight is "dangling", and its mass is spread
 *     uniformly rather than lost -- otherwise the scores stop summing to 1;
 *   * an undirected edge is walked in both directions.
 *
 * Teleportation can be aimed. By default the walk restarts anywhere with equal
 * probability, which measures importance in the graph as a whole; give it a
 * personalisation vector and it restarts only among the nodes you name, which
 * measures importance *relative to them*. That is the standard way to bias a
 * ranking towards a query -- seed the vector with the nodes a search matched
 * and the scores answer "what else matters, given these".
 */
final readonly class PageRank
{
    public function __construct(
        private float $damping = 0.85,
        private float $tolerance = 1.0e-12,
        private int $maxIterations = 1000,
    ) {
        if ($damping <= 0.0 || $damping >= 1.0) {
            throw InvalidArgument::outOfRange('Damping factor', $damping, 0.0, 1.0);
        }

        if ($tolerance <= 0.0) {
            throw InvalidArgument::outOfRange('Tolerance', $tolerance, PHP_FLOAT_EPSILON, INF);
        }
    }

    /**
     * @param  array<int, float>|null $personalisation node => weight; the walk
     *         teleports only to these, in proportion. Null spreads it evenly.
     * @return list<float> scores by node, summing to 1
     */
    public function of(Graph $graph, ?array $personalisation = null): array
    {
        $order = $graph->order();

        if ($order === 0) {
            return [];
        }

        $teleportTo = self::teleportDistribution($order, $personalisation);

        [$offsets, $targets, $weights] = $graph->csr();

        // Outgoing weight, self-loop counted once: this is the walk's view,
        // not the degree convention.
        $outgoing = array_fill(0, $order, 0.0);

        for ($node = 0; $node < $order; $node++) {
            $sum = 0.0;
            $end = $offsets[$node + 1];

            for ($i = $offsets[$node]; $i < $end; $i++) {
                $sum += $weights[$i];
            }

            $outgoing[$node] = $sum;
        }

        // The dangling nodes never change between iterations, so find them
        // once: the common case is none at all, and the per-iteration O(n)
        // scan for them was pure overhead.
        $danglingNodes = [];

        for ($node = 0; $node < $order; $node++) {
            if ($outgoing[$node] === 0.0) {
                $danglingNodes[] = $node;
            }
        }

        // Whether every stored weight is exactly 1.0. Not isWeighted(): that
        // flag reports the *input* -- two parallel 1.0 edges merge into a 2.0
        // that it does not see -- and the walk cares about the stored values.
        $unweighted = true;

        foreach ($weights as $weight) {
            if ($weight !== 1.0) {
                $unweighted = false;
                break;
            }
        }

        // Start from the teleport distribution rather than uniform: with a
        // personalisation vector that is where the walk lives, so starting
        // there converges sooner without changing the limit.
        $score = $teleportTo;

        $converged = false;
        $drift = INF;

        for ($iteration = 0; $iteration < $this->maxIterations; $iteration++) {
            $dangling = 0.0;

            foreach ($danglingNodes as $node) {
                $dangling += $score[$node];
            }

            // Teleporting mass plus mass stranded on dangling nodes. Both land
            // on the teleport distribution, not spread evenly -- otherwise a
            // personalised walk would leak into the rest of the graph and the
            // bias would wash out.
            $redistributed = (1.0 - $this->damping) + $this->damping * $dangling;
            $next = [];

            for ($node = 0; $node < $order; $node++) {
                $next[] = $redistributed * $teleportTo[$node];
            }

            for ($node = 0; $node < $order; $node++) {
                $out = $outgoing[$node];

                if ($out === 0.0) {
                    continue;
                }

                $share = $this->damping * $score[$node] / $out;
                $end = $offsets[$node + 1];

                if ($unweighted) {
                    for ($i = $offsets[$node]; $i < $end; $i++) {
                        $next[$targets[$i]] += $share;
                    }
                } else {
                    for ($i = $offsets[$node]; $i < $end; $i++) {
                        $next[$targets[$i]] += $share * $weights[$i];
                    }
                }
            }

            $drift = 0.0;

            for ($node = 0; $node < $order; $node++) {
                $drift += abs($next[$node] - $score[$node]);
            }

            $score = $next;

            // Scaled by order so the criterion means the same on any graph.
            if ($drift < $order * $this->tolerance) {
                $converged = true;

                break;
            }
        }

        if (! $converged) {
            throw DidNotConverge::after('PageRank', $this->maxIterations, $drift, $order * $this->tolerance);
        }

        /** @var list<float> $score */
        return $score;
    }

    /**
     * Normalise a personalisation into a distribution over every node.
     *
     * @param  array<int, float>|null $personalisation
     * @return list<float>
     */
    private static function teleportDistribution(int $order, ?array $personalisation): array
    {
        if ($personalisation === null) {
            return array_fill(0, $order, 1.0 / $order);
        }

        $weights = array_fill(0, $order, 0.0);
        $total = 0.0;

        foreach ($personalisation as $node => $weight) {
            if ($node < 0 || $node >= $order) {
                throw InvalidArgument::nodeOutOfRange($node, $order);
            }

            if ($weight < 0.0) {
                throw InvalidArgument::outOfRange(
                    "Personalisation weight for node {$node}",
                    $weight,
                    0.0,
                    INF,
                );
            }

            $weights[$node] += $weight;
            $total += $weight;
        }

        if ($total <= 0.0) {
            throw InvalidArgument::emptyDataset('a personalisation vector with no positive weight');
        }

        for ($node = 0; $node < $order; $node++) {
            $weights[$node] /= $total;
        }

        /** @var list<float> $weights */
        return $weights;
    }
}
