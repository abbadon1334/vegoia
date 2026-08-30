<?php

declare(strict_types=1);

namespace Vegoia\Graph\Community\Quality;

use function count;

use Vegoia\Graph\Graph;
use Vegoia\Graph\Partition;

/**
 * Surprise: how improbable this partition's internal edge count would be by
 * chance.
 *
 *     S = m * D( q || <q> )
 *
 * where q is the fraction of edges falling inside communities, <q> is the
 * fraction of *node pairs* that lie inside them, and D is the binary
 * Kullback-Leibler divergence. Read plainly: if edges were thrown down at
 * random, a fraction <q> of them would land inside communities; a partition is
 * surprising to the extent that many more than that actually did.
 *
 * Unlike modularity it has no resolution limit and no parameter to choose,
 * which is its appeal. Its bias runs the other way -- it favours many small
 * communities, and on a large sparse graph it will happily split further than
 * anyone wants.
 *
 * Defined on unweighted graphs: it counts edges and pairs, and there is no
 * agreed extension to weights. Weighted graphs are scored on their topology,
 * with weights ignored, which is what leidenalg does.
 *
 * @see V.A. Traag, R. Aldecoa & J.-C. Delvenne (2015), "Detecting communities
 *      using asymptotical surprise", Physical Review E 92, 022816.
 */
final class Surprise implements PartitionScore
{
    public function of(Graph $graph, Partition $partition): float
    {
        $order = $graph->order();
        $edges = 0;
        $internal = 0;

        foreach ($graph->edges() as [$from, $to]) {
            $edges++;

            if ($partition->communityOf($from) === $partition->communityOf($to)) {
                $internal++;
            }
        }

        if ($edges === 0 || $order < 2) {
            return 0.0;
        }

        $allPairs = $order * ($order - 1) / 2.0;
        $internalPairs = 0.0;

        foreach ($partition->communities() as $members) {
            $size = count($members);
            $internalPairs += $size * ($size - 1) / 2.0;
        }

        $observed = $internal / $edges;
        $expected = $internalPairs / $allPairs;

        return $edges * KullbackLeibler::binary($observed, $expected);
    }
}
