<?php

declare(strict_types=1);

namespace Vegoia\Graph;

use function count;
use function max;
use function min;

use Vegoia\Exception\InvalidArgument;
use Vegoia\Graph\Path\BreadthFirst;

/**
 * How far apart a graph's nodes are.
 *
 * A result object rather than four static functions, and that is a decision
 * about cost rather than about taste. Everything else in this namespace
 * memoises nothing, because most graph measures are one sweep for one answer
 * and the caller is expected to hold what they asked for. This sweep produces
 * four answers and it is the most expensive thing in the library: breadth-first
 * search from every node takes half a millisecond on 34 nodes, 2.6 on 77, and
 * 1.4 seconds on a thousand nodes and sixteen thousand edges. Four static
 * entry points would run it four times.
 *
 * Disconnected graphs are answered, not refused. NetworkX raises for diameter,
 * radius and average path length, and that is defensible -- but INF is the
 * answer rather than a guess. The diameter is a maximum over all pairs of a
 * shortest path length, some of those lengths are infinite, so the maximum is
 * infinite. The radius is infinite too, which is the part that surprises
 * people: every node fails to reach something, so no node has a finite
 * eccentricity over the whole graph. OneWayAnova already takes this position
 * for a ratio with no within-group variation, and BreadthFirst already takes
 * it for an unreachable node with its -1.
 *
 * What makes that useful rather than merely honest is that `$eccentricity` is
 * measured within each node's own component and is therefore always finite.
 * That costs nothing on a connected graph, where the component is the graph,
 * and on a disconnected one it is the only definition anybody can use -- the
 * largest component's diameter is `max($d->eccentricity)` over its members,
 * and its mean distance is `$d->totalDistance / $d->reachablePairs`.
 */
final readonly class Distance
{
    /** @param list<float> $eccentricity within each node's own component */
    private function __construct(
        /** @var list<float> */
        public array $eccentricity,
        public float $totalDistance,
        public int $reachablePairs,
        public int $components,
        public int $order,
    ) {
    }

    public static function of(Graph $graph): self
    {
        if ($graph->isDirected()) {
            throw InvalidArgument::directedNotSupported(
                'Distance measures',
                'breadth-first search follows arrows quite happily, which is the trap: the '
                . 'numbers would come out plausible while their finiteness had been judged '
                . 'against weak components, and the directed analogues are defined on strong '
                . 'connectivity instead.',
            );
        }

        $order = $graph->order();

        if ($order === 0) {
            return new self([], 0.0, 0, 0, 0);
        }

        $eccentricity = [];
        $total = 0.0;
        $pairs = 0;

        for ($node = 0; $node < $order; $node++) {
            $farthest = 0.0;

            foreach (BreadthFirst::distancesFrom($graph, $node) as $target => $hops) {
                // -1 marks unreachable. A node in another component says
                // nothing about this one's eccentricity and contributes no
                // pair, which is what keeps the eccentricity finite.
                if ($hops < 0.0 || $target === $node) {
                    continue;
                }

                $total += $hops;
                $pairs++;

                if ($hops > $farthest) {
                    $farthest = $hops;
                }
            }

            $eccentricity[] = $farthest;
        }

        /** @var list<float> $eccentricity */
        return new self(
            $eccentricity,
            $total,
            $pairs,
            Connectivity::components($graph)->count(),
            $order,
        );
    }

    /** The longest shortest path. INF when the graph is disconnected. */
    public function diameter(): float
    {
        if ($this->order === 0) {
            return 0.0;
        }

        /** @var non-empty-list<float> $eccentricity */
        $eccentricity = $this->eccentricity;

        return $this->isConnected() ? max($eccentricity) : INF;
    }

    /**
     * The smallest eccentricity. INF when the graph is disconnected -- every
     * node fails to reach something, so none of them has a finite one.
     */
    public function radius(): float
    {
        if ($this->order === 0) {
            return 0.0;
        }

        /** @var non-empty-list<float> $eccentricity */
        $eccentricity = $this->eccentricity;

        return $this->isConnected() ? min($eccentricity) : INF;
    }

    /**
     * The mean distance over every ordered pair that has one.
     *
     * Ordered pairs, so on an undirected graph each unordered pair is counted
     * twice in both the total and the count and the ratio is unaffected. Said
     * out loud because the other convention -- unordered pairs over
     * n(n-1)/2 -- is equally common and only agrees if both halves do.
     */
    public function averageShortestPathLength(): float
    {
        if (! $this->isConnected()) {
            return INF;
        }

        return $this->reachablePairs === 0 ? 0.0 : $this->totalDistance / $this->reachablePairs;
    }

    /** An empty graph is connected, agreeing with Connectivity::isConnected(). */
    public function isConnected(): bool
    {
        return $this->components <= 1;
    }
}
