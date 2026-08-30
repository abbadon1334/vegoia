<?php

declare(strict_types=1);

namespace Vegoia\Graph\Community\Quality;

use Vegoia\Graph\Graph;
use Vegoia\Graph\Partition;

/**
 * Something that scores a partition.
 *
 * Separate from QualityFunction because scoring and optimising are different
 * capabilities, and conflating them would be a lie the type system could not
 * catch. Modularity and CPM decompose into a per-community "weight minus
 * penalty" form, so moving one node changes the score by an expression in
 * local quantities and Leiden can hill-climb on them. Surprise and
 * Significance do not: both are divergences between global distributions, and
 * the effect of moving a node is only defined against totals over the whole
 * graph.
 *
 * So they implement this and not QualityFunction. You optimise with modularity
 * or CPM and score with Surprise -- which is how they are normally used
 * anyway, since their value is as an independent opinion on a partition
 * somebody else's objective produced.
 */
interface PartitionScore
{
    public function of(Graph $graph, Partition $partition): float;
}
