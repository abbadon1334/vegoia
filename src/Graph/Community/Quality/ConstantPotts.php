<?php

declare(strict_types=1);

namespace Vegoia\Graph\Community\Quality;

use function count;

use Vegoia\Exception\InvalidArgument;
use Vegoia\Graph\Graph;
use Vegoia\Graph\Partition;
use Vegoia\Support\CompensatedSum;

/**
 * The Constant Potts Model.
 *
 *     H = sum_c [ e_c - y * n_c (n_c - 1) / 2 ]
 *
 * For each community: the weight actually inside it, minus the resolution
 * times the number of pairs it could have contained. A community is worth
 * keeping when its internal density beats `y`, and that judgement is made
 * community by community -- nothing in the formula refers to the size of the
 * graph.
 *
 * That is the whole point. Modularity compares against a global null model, so
 * what counts as a community depends on how much graph surrounds it, and small
 * communities vanish in large graphs. CPM has no such term, so `y` means the
 * same thing at every scale: it is a density threshold, and communities are
 * guaranteed to be at least that dense and separated by less than that.
 *
 * The trade is that `y` must be chosen rather than defaulted. Sweeping it and
 * watching where the partition is stable is the usual way in.
 *
 * @see V.A. Traag, P. Van Dooren & Y. Nesterov (2011), Physical Review E 84, 016114.
 */
final readonly class ConstantPotts implements QualityFunction
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

                    // Halved below, so count each internal edge from both ends;
                    // a self-loop is stored once and is its own pair.
                    $internalWeight += $neighbour === $node ? 2.0 * $weight : $weight;
                }
            }

            $size = (float) count($members);

            $total->add($internalWeight / 2.0 - $this->resolution * $size * ($size - 1.0) / 2.0);
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
        return $weightToCommunity - $this->resolution * $nodeSize * $communitySize;
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
        return $this->resolution * $partMeasure * ($subsetMeasure - $partMeasure);
    }
}
