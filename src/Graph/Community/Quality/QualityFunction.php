<?php

declare(strict_types=1);

namespace Vegoia\Graph\Community\Quality;

use Vegoia\Graph\Graph;
use Vegoia\Graph\Partition;

/**
 * The objective a community-detection run is trying to maximise.
 *
 * Leiden is indifferent to which objective it optimises, so the objective is
 * an interface rather than a branch inside the algorithm. Everything the
 * search needs is expressed here in four operations, all of which must be O(1)
 * because the inner loop calls them once per candidate community per node.
 */
interface QualityFunction
{
    public function resolution(): float;

    /** The objective's value for a whole partition. */
    public function of(Graph $graph, Partition $partition): float;

    /**
     * How much better the objective gets when a node joins a community.
     *
     * Proportional to the true delta, not equal to it: for modularity the
     * common factor 2/(2m) is positive and identical for every candidate, so
     * dropping it changes no comparison while saving a division in the hottest
     * loop in the library.
     *
     * @param float $weightToCommunity edge weight from the node into the community
     * @param float $communityStrength total strength of the community, node excluded
     * @param float $communitySize     total size of the community, node excluded
     */
    public function gain(
        float $weightToCommunity,
        float $nodeStrength,
        float $nodeSize,
        float $communityStrength,
        float $communitySize,
        float $totalEndpointWeight,
    ): float;

    /**
     * The quantity this objective measures a set by: strength for modularity,
     * plain node count for CPM. The refinement phase needs it to decide which
     * parts are well connected to their surroundings.
     */
    public function measure(float $strength, float $size): float;

    /**
     * The connectivity a part must have to the rest of its subset before
     * refinement will consider merging into it -- the gamma-connectivity
     * condition of the Leiden paper. Below this threshold a part is left
     * alone, and that is precisely what stops a community from being welded
     * together through a node that does not belong to it.
     */
    public function connectivityThreshold(
        float $partMeasure,
        float $subsetMeasure,
        float $totalEndpointWeight,
    ): float;
}
