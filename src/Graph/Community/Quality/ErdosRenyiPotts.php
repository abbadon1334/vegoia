<?php

declare(strict_types=1);

namespace Vegoia\Graph\Community\Quality;

use function count;

use Vegoia\Exception\InvalidArgument;
use Vegoia\Graph\Graph;
use Vegoia\Graph\Partition;
use Vegoia\Support\CompensatedSum;

/**
 * Reichardt-Bornholdt with an Erdos-Renyi null model -- RBER in leidenalg.
 *
 *     H = sum_c [ e_c - y * p * n_c (n_c - 1) / 2 ]
 *
 * Constant Potts with the resolution scaled by the graph's own density p.
 * That single change moves it between the two families: CPM's `y` is an
 * absolute density threshold that means the same on any graph, while here it
 * is relative to how dense this graph happens to be, so y = 1 asks for
 * communities denser than the graph average whatever that average is.
 *
 * Useful when comparing partitions across graphs of different densities,
 * where CPM's absolute threshold would be measuring the density difference
 * rather than the community structure. It keeps CPM's per-community
 * judgement, so unlike modularity it has no resolution limit.
 *
 * @see J. Reichardt & S. Bornholdt (2006), "Statistical mechanics of community
 *      detection", Physical Review E 74, 016110.
 */
final readonly class ErdosRenyiPotts implements QualityFunction
{
    /**
     * @param float|null $density the graph's density, once known. Until then
     *                            of() computes it per call and gain() cannot
     *                            be used -- see boundTo().
     */
    public function __construct(
        private float $resolution = 1.0,
        private ?float $density = null,
    ) {
        if ($resolution < 0.0) {
            throw InvalidArgument::outOfRange('Resolution', $resolution, 0.0, INF);
        }
    }

    public function resolution(): float
    {
        return $this->resolution;
    }

    /** The bound density, or 1.0 when this instance has not seen a graph. */
    public function density(): float
    {
        return $this->density ?? 1.0;
    }

    public function boundTo(Graph $graph): self
    {
        return new self($this->resolution, self::densityOf($graph));
    }

    /**
     * The graph's edge density, which scales the penalty.
     *
     * Weighted graphs use total weight rather than edge count, matching
     * leidenalg: the null model is "this much weight spread evenly over all
     * possible pairs".
     */
    public static function densityOf(Graph $graph): float
    {
        $order = $graph->order();

        if ($order < 2) {
            return 0.0;
        }

        return $graph->totalWeight() / ($order * ($order - 1) / 2.0);
    }

    public function of(Graph $graph, Partition $partition): float
    {
        // Prefer the graph in front of us over a bound value: scoring a
        // different graph with a stale density would silently misreport.
        $density = self::densityOf($graph);
        $total = new CompensatedSum();

        foreach ($partition->communities() as $members) {
            $inside = [];
            foreach ($members as $node) {
                $inside[$node] = true;
            }

            $internalWeight = 0.0;

            foreach ($members as $node) {
                foreach ($graph->neighbours($node) as $neighbour => $weight) {
                    if (! isset($inside[$neighbour])) {
                        continue;
                    }

                    $internalWeight += $neighbour === $node ? 2.0 * $weight : $weight;
                }
            }

            $size = (float) count($members);

            $total->add(
                $internalWeight / 2.0 - $this->resolution * $density * $size * ($size - 1.0) / 2.0
            );
        }

        return $total->value();
    }

    public function gain(
        float $weightToCommunity,
        float $nodeStrength,
        float $nodeSize,
        float $communityStrength,
        float $communitySize,
        float $totalEndpointWeight,
    ): float {
        if ($this->density === null) {
            throw InvalidArgument::malformedEdge(
                'ErdosRenyiPotts needs the graph density before it can compute a gain. '
                . 'Call boundTo($graph) first; Leiden does this for you.'
            );
        }

        return $weightToCommunity - $this->resolution * $this->density * $nodeSize * $communitySize;
    }

    public function measure(float $strength, float $size): float
    {
        return $size;
    }

    public function connectivityThreshold(
        float $partMeasure,
        float $subsetMeasure,
        float $totalEndpointWeight,
    ): float {
        return $this->resolution * ($this->density ?? 1.0) * $partMeasure * ($subsetMeasure - $partMeasure);
    }
}
