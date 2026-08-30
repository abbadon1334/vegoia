<?php

declare(strict_types=1);

namespace Vegoia\Graph\Community\Quality;

use function count;

use Vegoia\Graph\Graph;
use Vegoia\Graph\Partition;
use Vegoia\Support\CompensatedSum;

/**
 * Significance: how improbably dense each community is, summed.
 *
 *     S = sum_c  C(n_c, 2) * D( p_c || p )
 *
 * where p_c is the community's internal density, p the graph's, and D the
 * binary Kullback-Leibler divergence. Each community contributes the
 * improbability of its own density weighted by how many pairs it contains, so
 * a small very dense group and a large mildly dense one can score alike.
 *
 * Parameter-free, like Surprise, and with the same restriction to unweighted
 * graphs. Where they differ: Surprise asks one global question about all
 * internal edges at once, while Significance asks it per community and adds
 * up -- so Significance notices a single anomalously dense group that Surprise
 * would average away.
 *
 * @see V.A. Traag, G. Krings & P. Van Dooren (2013), "Significant scales in
 *      community structure", Scientific Reports 3, 2930.
 */
final class Significance implements PartitionScore
{
    public function of(Graph $graph, Partition $partition): float
    {
        $order = $graph->order();

        if ($order < 2) {
            return 0.0;
        }

        $edges = 0;
        $internalOf = [];

        foreach ($graph->edges() as [$from, $to]) {
            $edges++;
            $community = $partition->communityOf($from);

            if ($community === $partition->communityOf($to)) {
                $internalOf[$community] = ($internalOf[$community] ?? 0) + 1;
            }
        }

        $density = $edges / ($order * ($order - 1) / 2.0);
        $total = new CompensatedSum();

        foreach ($partition->communities() as $community => $members) {
            $size = count($members);

            if ($size < 2) {
                continue;
            }

            $pairs = $size * ($size - 1) / 2.0;
            $internalDensity = ($internalOf[$community] ?? 0) / $pairs;

            $total->add($pairs * KullbackLeibler::binary($internalDensity, $density));
        }

        return $total->value();
    }
}
