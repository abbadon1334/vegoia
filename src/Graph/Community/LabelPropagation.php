<?php

declare(strict_types=1);

namespace Vegoia\Graph\Community;

use function count;

use Random\Engine\Xoshiro256StarStar;
use Random\Randomizer;

use function range;

use Vegoia\Exception\InvalidArgument;
use Vegoia\Graph\Graph;
use Vegoia\Graph\Partition;

/**
 * Communities by label propagation (Raghavan, Albert and Kumara, 2007).
 *
 * The cheap one. Every node starts as its own community and repeatedly adopts
 * whichever label carries the most weight among its neighbours; that is the
 * entire algorithm. It runs in time linear in the number of edges, has no
 * objective function, no resolution parameter, and nothing to tune.
 *
 * It is worth having next to Leiden precisely because it shares none of
 * Leiden's assumptions. Leiden maximises a quality function, and every quality
 * function encodes a belief about what a community is -- modularity's null
 * model, CPM's density threshold. Label propagation encodes almost nothing: it
 * finds the groups that agree with themselves. When the two disagree about a
 * graph, that disagreement is information.
 *
 * The price is instability, and it is not a small one. With nothing being
 * maximised there is no sense in which one run is better than another, and on
 * a graph without clear structure the algorithm will happily collapse
 * everything into one community or leave everything apart -- on the Petersen
 * graph, NetworkX's own implementation returns anywhere from one community to
 * four across fifty seeds. Use it on large graphs with real structure, or as
 * a second opinion. Use Leiden when the answer matters.
 *
 * Asynchronous: a node sees the labels its neighbours took this round, not the
 * ones they held at the start of it. Synchronous updating is simpler and
 * oscillates forever on bipartite structures, two halves swapping labels every
 * round without ever settling.
 */
final class LabelPropagation
{
    public function __construct(
        private readonly int $seed = 0,
        private readonly int $maxIterations = 100,
    ) {
        if ($maxIterations < 1) {
            throw InvalidArgument::outOfRange('Maximum iterations', (float) $maxIterations, 1.0, INF);
        }
    }

    public function partition(Graph $graph, ?Randomizer $randomizer = null): Partition
    {
        $order = $graph->order();

        if ($order === 0) {
            return Partition::fromMembership([]);
        }

        $randomizer ??= new Randomizer(new Xoshiro256StarStar($this->seed));

        [$offsets, $targets, $weights] = $graph->csr();

        // Every node its own label. Isolated nodes keep theirs, which is
        // right: a node with no neighbours has nothing to agree with.
        /** @var list<int> $labels */
        $labels = range(0, $order - 1);

        for ($iteration = 0; $iteration < $this->maxIterations; $iteration++) {
            /** @var list<int> $visiting */
            $visiting = $randomizer->shuffleArray(range(0, $order - 1));
            $changed = false;

            foreach ($visiting as $node) {
                $best = self::heaviestLabels($node, $labels, $offsets, $targets, $weights);

                // A node that already holds one of the heaviest labels keeps
                // it. Reassigning at random among equals instead -- which the
                // first draft of this did -- churns for no reason and
                // cascades: on a path of ten nodes it collapsed the whole
                // graph into one community on seeds where the reference never
                // does. It also removes the stopping condition, since a node
                // swapping between two equally good labels for ever is not
                // progress.
                if ($best === [] || in_array($labels[$node], $best, strict: true)) {
                    continue;
                }

                /** @var list<int> $shuffled */
                $shuffled = count($best) === 1 ? $best : $randomizer->shuffleArray($best);
                /** @var list<int> $labels */
                $labels[$node] = $shuffled[0];
                $changed = true;
            }

            if (! $changed) {
                break;
            }
        }

        return Partition::fromMembership(self::compact($labels));
    }

    /**
     * Every label tying for the most weight among a node's neighbours.
     *
     * Empty when the node has no neighbours to consult, which is not a
     * degenerate case: an isolated node keeps the label it started with,
     * because it has nothing to agree with.
     *
     * @param list<int>   $labels
     * @param list<int>   $offsets
     * @param list<int>   $targets
     * @param list<float> $weights
     *
     * @return list<int>
     */
    private static function heaviestLabels(
        int $node,
        array $labels,
        array $offsets,
        array $targets,
        array $weights,
    ): array {
        $tally = [];
        $end = $offsets[$node + 1];

        for ($i = $offsets[$node]; $i < $end; $i++) {
            $neighbour = $targets[$i];

            // A self-loop says a node agrees with itself, which is true of
            // every node and so tells the algorithm nothing.
            if ($neighbour === $node) {
                continue;
            }

            $label = $labels[$neighbour];
            $tally[$label] = ($tally[$label] ?? 0.0) + $weights[$i];
        }

        if ($tally === []) {
            return [];
        }

        $heaviest = max($tally);
        $best = [];

        foreach ($tally as $label => $weight) {
            if ($weight === $heaviest) {
                $best[] = $label;
            }
        }

        return $best;
    }

    /**
     * Renumber surviving labels to 0..k-1 in order of first appearance.
     *
     * Partition requires contiguous community numbers, and label propagation
     * leaves whichever of the original node numbers happened to win.
     *
     * @param list<int> $labels
     *
     * @return list<int>
     */
    private static function compact(array $labels): array
    {
        $renumbered = [];
        $membership = [];

        foreach ($labels as $label) {
            $membership[] = $renumbered[$label] ??= count($renumbered);
        }

        return $membership;
    }
}
