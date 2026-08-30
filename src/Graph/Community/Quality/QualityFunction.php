<?php

declare(strict_types=1);

namespace Vegoia\Graph\Community\Quality;

use Vegoia\Graph\Graph;

/**
 * An objective a community-detection run can maximise.
 *
 * Leiden is indifferent to which objective it optimises, so the objective is
 * an interface rather than a branch inside the algorithm. Everything the
 * search needs is expressed here in four operations, all of which must be O(1)
 * because the inner loop calls them once per candidate community per node.
 */
interface QualityFunction extends PartitionScore
{
    public function resolution(): float;

    /**
     * Bind whatever this objective needs from the graph before optimisation.
     *
     * gain() is given only local quantities, because passing the graph into a
     * call made millions of times per run would be absurd. An objective whose
     * formula involves a graph-level constant -- ErdosRenyiPotts needs the
     * density -- therefore cannot compute its own gain until it has seen the
     * graph. This is where it gets to look, once, and return an instance that
     * can.
     *
     * Objectives with no such constant return themselves, which is the
     * default and costs nothing.
     */
    public function boundTo(Graph $graph): self;

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
