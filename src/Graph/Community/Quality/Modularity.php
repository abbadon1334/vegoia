<?php

declare(strict_types=1);

namespace Vegoia\Graph\Community\Quality;

use Vegoia\Exception\InvalidArgument;
use Vegoia\Graph\Graph;
use Vegoia\Graph\Partition;
use Vegoia\Support\CompensatedSum;

/**
 * Newman-Girvan modularity with a resolution parameter.
 *
 *     Q = (1 / 2m) * sum_ij [ A_ij - y * k_i k_j / 2m ] * d(c_i, c_j)
 *
 * Read it as a comparison, because that is what it is: the first term counts
 * the edges actually inside communities, the second counts how many a random
 * graph with the same degree sequence would have put there. Structure is
 * whatever the null model fails to explain.
 *
 * Known limitation, not a defect: modularity has a resolution limit. Below a
 * scale set by the total edge weight it cannot separate small communities at
 * all, and will merge adjacent ones however clearly they are divided -- the
 * `ring_of_cliques` fixture exists to keep that fact visible. When the
 * communities you care about are small relative to the graph, ConstantPotts
 * is the objective that does not have this failure mode.
 *
 * @see M.E.J. Newman & M. Girvan (2004), Physical Review E 69, 026113.
 * @see S. Fortunato & M. Barthelemy (2007), PNAS 104(1), 36-41.
 */
final readonly class Modularity implements QualityFunction
{
    public function __construct(private float $resolution = 1.0)
    {
        if ($resolution < 0.0) {
            throw InvalidArgument::outOfRange('Resolution', $resolution, 0.0, INF);
        }
    }

    public function resolution(): float
    {
        return $this->resolution;
    }

    public function of(Graph $graph, Partition $partition): float
    {
        $twoM = $graph->totalEndpointWeight();

        if ($twoM === 0.0) {
            return 0.0;
        }

        $internal = new CompensatedSum();
        $expected = new CompensatedSum();

        foreach ($partition->communities() as $members) {
            $inside = [];
            foreach ($members as $node) {
                $inside[$node] = true;
            }

            $internalWeight = 0.0;
            $strength = 0.0;

            foreach ($members as $node) {
                $strength += $graph->strength($node);

                foreach ($graph->neighbours($node) as $neighbour => $weight) {
                    if (! isset($inside[$neighbour])) {
                        continue;
                    }

                    // Each internal edge is seen from both ends, so half of the
                    // running total is the edge weight; a self-loop is seen once
                    // but joins the node to itself twice, so it contributes the
                    // same way once doubled.
                    $internalWeight += $neighbour === $node ? 2.0 * $weight : $weight;
                }
            }

            $internal->add($internalWeight / $twoM);
            $expected->add($this->resolution * ($strength / $twoM) ** 2);
        }

        return $internal->value() - $expected->value();
    }

    public function gain(
        float $weightToCommunity,
        float $nodeStrength,
        float $nodeSize,
        float $communityStrength,
        float $communitySize,
        float $totalEndpointWeight,
    ): float {
        if ($totalEndpointWeight === 0.0) {
            return 0.0;
        }

        return $weightToCommunity
            - $this->resolution * $nodeStrength * $communityStrength / $totalEndpointWeight;
    }

    public function measure(float $strength, float $size): float
    {
        return $strength;
    }

    public function connectivityThreshold(
        float $partMeasure,
        float $subsetMeasure,
        float $totalEndpointWeight,
    ): float {
        if ($totalEndpointWeight === 0.0) {
            return 0.0;
        }

        return $this->resolution * $partMeasure * ($subsetMeasure - $partMeasure) / $totalEndpointWeight;
    }
}
