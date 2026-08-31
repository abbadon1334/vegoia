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
    /**
     * Edges are walked over the raw CSR arrays rather than through the
     * edges() generator. That is not only for speed: under the PHP 8.5.10
     * tracing JIT this loop, consumed through the generator, counted 105 edges
     * on a 78-edge graph -- and produced a Surprise of 47.18 where the correct
     * value is 37.62. The same counting over the arrays is right under both
     * the interpreter and the JIT. The generator remains correct for ordinary
     * use; it is the compiled hot path that goes wrong, so the hot paths avoid
     * it.
     */
    public function of(Graph $graph, Partition $partition): float
    {
        $order = $graph->order();
        [$offsets, $targets] = $graph->csr();
        $membership = $partition->membership();
        $directed = $graph->isDirected();

        $edges = 0;
        $internal = 0;

        for ($from = 0; $from < $order; $from++) {
            $end = $offsets[$from + 1];

            for ($i = $offsets[$from]; $i < $end; $i++) {
                $to = $targets[$i];

                // Undirected storage holds both directions; count each once.
                if (! $directed && $to < $from) {
                    continue;
                }

                $edges++;

                if ($membership[$from] === $membership[$to]) {
                    $internal++;
                }
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
