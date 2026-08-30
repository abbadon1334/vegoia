<?php

declare(strict_types=1);

namespace Vegoia\Graph\Community;

use function abs;
use function array_fill;
use function array_pop;
use function count;
use function exp;
use function intdiv;
use function max;

use Random\Engine\Xoshiro256StarStar;
use Random\Randomizer;

use function range;

use Vegoia\Exception\InvalidArgument;
use Vegoia\Graph\Community\Quality\ConstantPotts;
use Vegoia\Graph\Community\Quality\Modularity;
use Vegoia\Graph\Community\Quality\QualityFunction;
use Vegoia\Graph\Graph;
use Vegoia\Graph\Partition;

/**
 * The Leiden algorithm for community detection.
 *
 * Louvain, its predecessor, has a defect that is easy to miss and hard to live
 * with: it can return a community that is internally disconnected. A node acts
 * as the only bridge holding a community together, a later move takes it
 * elsewhere, and nothing ever revisits the pieces it left behind. The result
 * still scores well, so nothing complains -- you simply get "communities" whose
 * members are not connected to each other. In a retrieval pipeline that is a
 * summary of two unrelated topics presented as one.
 *
 * Leiden fixes this by inserting a refinement phase between local moving and
 * aggregation. Instead of collapsing each community wholesale, it first splits
 * each one into well-connected parts and aggregates *those*. A community can
 * therefore be broken apart again on a later pass, and the algorithm
 * guarantees -- not merely tends towards -- internally connected communities.
 *
 * The three phases:
 *
 *   1. local moving: visit nodes in random order, move each to the community
 *      that improves the objective most, and re-queue the neighbours a move
 *      might have unsettled. Only nodes that could have changed are revisited.
 *   2. refinement: within each community, merge nodes into well-connected
 *      subsets, choosing among improving merges at random rather than greedily.
 *      That randomness is what lets the partition space actually get explored.
 *   3. aggregation: collapse each refined subset into one node and repeat on
 *      the smaller graph, until nothing merges.
 *
 * Runs are reproducible: the same seed gives the same partition. Different
 * seeds give different, equally valid partitions -- community detection is an
 * optimisation over a rugged landscape, not a function with one answer.
 *
 * @see V.A. Traag, L. Waltman & N.J. van Eck (2019), "From Louvain to Leiden:
 *      guaranteeing well-connected communities", Scientific Reports 9, 5233.
 */
final class Leiden
{
    /**
     * How much the refinement phase is willing to take a worse improving move.
     * Interpreted relative to the best gain available, so it means the same
     * thing on a graph of 10 edges and a graph of 10 million.
     */
    public const float DEFAULT_RANDOMNESS = 0.01;

    /** Aggregation shrinks the graph fast; convergence normally arrives well before this. */
    public const int DEFAULT_MAX_ITERATIONS = 32;

    public function __construct(
        private readonly QualityFunction $objective = new Modularity(),
        private readonly int $seed = 0,
        private readonly float $randomness = self::DEFAULT_RANDOMNESS,
        private readonly int $maxIterations = self::DEFAULT_MAX_ITERATIONS,
    ) {
        if ($randomness <= 0.0) {
            throw InvalidArgument::outOfRange('Randomness', $randomness, PHP_FLOAT_EPSILON, INF);
        }

        if ($maxIterations < 1) {
            throw InvalidArgument::outOfRange('Maximum iterations', (float) $maxIterations, 1.0, INF);
        }
    }

    /** Optimise modularity: the usual default when you have no scale in mind. */
    public static function modularity(float $resolution = 1.0, int $seed = 0): self
    {
        return new self(new Modularity($resolution), $seed);
    }

    /**
     * Optimise the Constant Potts Model: use this when the communities you care
     * about are small relative to the graph, where modularity's resolution
     * limit would merge them.
     */
    public static function constantPotts(float $resolution = 1.0, int $seed = 0): self
    {
        return new self(new ConstantPotts($resolution), $seed);
    }

    public function objective(): QualityFunction
    {
        return $this->objective;
    }

    /**
     * 1 for Modularity, 2 for ConstantPotts, 0 for anything else.
     *
     * The gain of both built-in objectives is a single multiply-subtract, and
     * the inner loops evaluate it once per candidate community of every node
     * on every pass -- through the interface, that is an interpreted method
     * call in the hottest loop of the library, and it dominates the profile.
     * The exact class is compared (not instanceof: a subclass may override
     * the formula) to select an inlined arithmetic path; any other objective
     * keeps the interface path and stays correct, just slower.
     */
    private function fastMode(): int
    {
        return match ($this->objective::class) {
            Modularity::class => 1,
            ConstantPotts::class => 2,
            default => 0,
        };
    }

    /**
     * The shared factor of both inlined gains: modularity's proportional gain
     * is w - (y/2m) k_i K_c, CPM's is w - y n_i N_c. The same factor also
     * appears in the gamma-connectivity threshold, y a (S - a) [/ 2m].
     */
    private function fastFactor(int $mode, float $totalEndpointWeight): float
    {
        if ($mode === 1) {
            return $totalEndpointWeight > 0.0
                ? $this->objective->resolution() / $totalEndpointWeight
                : 0.0;
        }

        return $this->objective->resolution();
    }

    public function partition(Graph $graph): Partition
    {
        return $this->partitionWithTrace($graph)[0];
    }

    /**
     * The partition, plus what the algorithm did to reach it.
     *
     * One entry per aggregation level: the size of the graph at that level,
     * how many communities local moving found, and how many parts refinement
     * split them into. `refined` above `communities` is the whole difference
     * from Louvain -- it is the number of nodes the next level will have, and
     * where it equals `communities` refinement found nothing to split, so the
     * level behaved exactly as Louvain would.
     *
     * Useful for tuning -- a resolution that collapses everything at level 0
     * is visible immediately -- and it is what makes the refinement phase
     * observable to a test. Without it, an implementation that skipped
     * refinement entirely still passed every assertion in this library, since
     * on graphs whose structure is clear enough Louvain finds the same answer.
     *
     * Each entry also carries the level's graph and both partitions. They are
     * objects that already exist at that point, so this costs a reference and
     * not a computation, and it lets a caller -- or a test -- check the
     * gamma-connectivity condition refinement is supposed to enforce, which is
     * otherwise invisible from outside.
     *
     * @return array{Partition, list<array{level: int, nodes: int, communities: int, refined: int, graph: Graph, partition: Partition, parts: Partition}>}
     */
    public function partitionWithTrace(Graph $graph): array
    {
        $order = $graph->order();

        if ($order === 0) {
            return [Partition::fromMembership([]), []];
        }

        $trace = [];
        $randomizer = new Randomizer(new Xoshiro256StarStar($this->seed));
        $totalEndpointWeight = $graph->totalEndpointWeight();

        $current = $graph;
        $sizes = array_fill(0, $order, 1.0);
        $membership = range(0, $order - 1);

        // Where each original node currently lives, as the graph collapses.
        $mapping = range(0, $order - 1);

        for ($iteration = 0; $iteration < $this->maxIterations; $iteration++) {
            $membership = $this->moveNodes($current, $membership, $sizes, $totalEndpointWeight, $randomizer);
            $outer = Partition::fromMembership($membership);

            // One node per community means aggregation would change nothing.
            if ($outer->count() === $current->order()) {
                $membership = $outer->membership();
                break;
            }

            $refined = $this->refine($current, $outer, $sizes, $totalEndpointWeight, $randomizer);

            $trace[] = [
                'level' => $iteration,
                'nodes' => $current->order(),
                'communities' => $outer->count(),
                'refined' => $refined->count(),
                'graph' => $current,
                'partition' => $outer,
                'parts' => $refined,
            ];

            [$aggregated, $aggregatedSizes, $induced] = $this->aggregate($current, $refined, $outer, $sizes);

            $refinedMembership = $refined->membership();

            foreach ($mapping as $original => $node) {
                $mapping[$original] = $refinedMembership[$node];
            }

            $current = $aggregated;
            $sizes = $aggregatedSizes;
            $membership = $induced;
        }

        $final = Partition::fromMembership($membership);
        $finalMembership = $final->membership();

        $result = [];
        foreach ($mapping as $node) {
            $result[] = $finalMembership[$node];
        }

        return [Partition::fromMembership($result), $trace];
    }

    /**
     * Phase 1. A queue rather than repeated full sweeps: moving a node can
     * only change the best choice for its neighbours, so only they are worth
     * re-examining. On a sparse graph that turns a quadratic scan into work
     * proportional to the moves actually made.
     *
     * @param  list<int>   $membership
     * @param  list<float> $sizes
     * @return list<int>
     */
    private function moveNodes(
        Graph $graph,
        array $membership,
        array $sizes,
        float $totalEndpointWeight,
        Randomizer $randomizer,
    ): array {
        $order = $graph->order();
        [$offsets, $targets, $weights] = $graph->csr();
        $strengths = $graph->strengths();

        $communityStrength = array_fill(0, $order, 0.0);
        $communitySize = array_fill(0, $order, 0.0);

        foreach ($membership as $node => $community) {
            $communityStrength[$community] += $strengths[$node];
            $communitySize[$community] += $sizes[$node];
        }

        /** @var list<int> $queue */
        $queue = $randomizer->shuffleArray(range(0, $order - 1));
        $queued = array_fill(0, $order, true);
        $head = 0;
        $length = $order;

        /** @var list<int> $vacated community indices known to be empty */
        $vacated = [];

        $mode = $this->fastMode();
        $isModularity = $mode === 1;
        $factor = $this->fastFactor($mode, $totalEndpointWeight);

        // NOTE on an optimisation deliberately not taken: replacing the small
        // per-node $toCommunity hash below with flat reusable buffers plus a
        // version stamp looks like an obvious win, and is not. Measured on PHP
        // 8.5.10 it was slower under the interpreter, and under the tracing
        // JIT it produced a *different, worse* partition than unjitted runs of
        // the same seed (Q dropped from 0.891 to 0.854 on a 100k-node graph)
        // -- a divergence this library cannot accept, since a seed is promised
        // to be reproducible. The hash version is JIT-stable.
        while ($head < $length) {
            $node = $queue[$head++];
            $queued[$node] = false;

            $origin = $membership[$node];
            $strength = $strengths[$node];
            $size = $sizes[$node];

            $communityStrength[$origin] -= $strength;
            $communitySize[$origin] -= $size;
            $originEmptied = $communitySize[$origin] <= 0.0;

            $end = $offsets[$node + 1];

            /** @var array<int, float> $toCommunity */
            $toCommunity = [];

            for ($i = $offsets[$node]; $i < $end; $i++) {
                $neighbour = $targets[$i];

                if ($neighbour === $node) {
                    continue;
                }

                $community = $membership[$neighbour];
                $toCommunity[$community] = ($toCommunity[$community] ?? 0.0) + $weights[$i];
            }

            // Staying put is the baseline. When the origin just emptied, this
            // is also the "become a singleton" option, and it scores zero.
            $best = $origin;

            if ($mode !== 0) {
                // Inlined gain: one multiply-subtract per candidate. The
                // branch on the mode selects which pair of running totals
                // feeds it; aliasing the array instead would trigger a full
                // copy-on-write when the totals are updated after the move.
                $nodeFactor = $factor * ($isModularity ? $strength : $size);
                $bestGain = ($toCommunity[$origin] ?? 0.0)
                    - $nodeFactor * ($isModularity ? $communityStrength[$origin] : $communitySize[$origin]);

                foreach ($toCommunity as $community => $weight) {
                    if ($community === $origin) {
                        continue;
                    }

                    $gain = $weight
                        - $nodeFactor * ($isModularity ? $communityStrength[$community] : $communitySize[$community]);

                    if ($gain > $bestGain) {
                        $bestGain = $gain;
                        $best = $community;
                    }
                }
            } else {
                $bestGain = $this->objective->gain(
                    $toCommunity[$origin] ?? 0.0,
                    $strength,
                    $size,
                    $communityStrength[$origin],
                    $communitySize[$origin],
                    $totalEndpointWeight,
                );

                foreach ($toCommunity as $community => $weight) {
                    if ($community === $origin) {
                        continue;
                    }

                    $gain = $this->objective->gain(
                        $weight,
                        $strength,
                        $size,
                        $communityStrength[$community],
                        $communitySize[$community],
                        $totalEndpointWeight,
                    );

                    if ($gain > $bestGain) {
                        $bestGain = $gain;
                        $best = $community;
                    }
                }
            }

            // Every attachment makes things worse, so leave: an empty community
            // scores exactly zero, which beats any negative gain.
            if ($bestGain < 0.0 && ! $originEmptied) {
                /** @var list<float> $communitySize */
                $slot = $this->takeVacatedSlot($vacated, $communitySize);

                if ($slot !== null) {
                    $best = $slot;
                    $bestGain = 0.0;
                }
            }

            $membership[$node] = $best;
            $communityStrength[$best] += $strength;
            $communitySize[$best] += $size;

            if ($best === $origin) {
                continue;
            }

            if ($originEmptied) {
                $vacated[] = $origin;
            }

            // A move can only unsettle neighbours that ended up elsewhere.
            for ($i = $offsets[$node]; $i < $end; $i++) {
                $neighbour = $targets[$i];

                if ($neighbour === $node || $queued[$neighbour] || $membership[$neighbour] === $best) {
                    continue;
                }

                $queued[$neighbour] = true;
                $queue[$length++] = $neighbour;
            }
        }

        /** @var list<int> $membership keys are untouched; only values change */
        return $membership;
    }

    /**
     * Phase 2, the phase Louvain lacks.
     *
     * Every community from phase 1 is re-partitioned from singletons, and only
     * subsets that are well connected to the rest of their community are
     * allowed to merge. Aggregating these refined parts rather than whole
     * communities is what leaves a badly-connected community able to come apart
     * on the next pass.
     *
     * @param list<float> $sizes
     */
    private function refine(
        Graph $graph,
        Partition $outer,
        array $sizes,
        float $totalEndpointWeight,
        Randomizer $randomizer,
    ): Partition {
        $order = $graph->order();
        [$offsets, $targets, $weights] = $graph->csr();
        $strengths = $graph->strengths();

        $membership = range(0, $order - 1);
        $communityStrength = $strengths;
        $communitySize = $sizes;
        $memberCount = array_fill(0, $order, 1);

        // Weight from each refined community out to the rest of its subset.
        $external = array_fill(0, $order, 0.0);

        $mode = $this->fastMode();
        $isModularity = $mode === 1;
        $factor = $this->fastFactor($mode, $totalEndpointWeight);

        foreach ($outer->communities() as $subset) {
            if (count($subset) <= 1) {
                continue;
            }

            $inside = [];
            $subsetMeasure = 0.0;

            foreach ($subset as $node) {
                $inside[$node] = true;
                $subsetMeasure += $mode !== 0
                    ? ($isModularity ? $strengths[$node] : $sizes[$node])
                    : $this->objective->measure($strengths[$node], $sizes[$node]);
            }

            foreach ($subset as $node) {
                $weight = 0.0;
                $end = $offsets[$node + 1];

                for ($i = $offsets[$node]; $i < $end; $i++) {
                    $neighbour = $targets[$i];

                    if ($neighbour !== $node && isset($inside[$neighbour])) {
                        $weight += $weights[$i];
                    }
                }

                $external[$node] = $weight;
            }

            // Only nodes already well connected to their subset may merge; a
            // node hanging off the edge of a community stays a singleton and so
            // can be pulled away entirely on a later pass.
            $candidates = [];

            foreach ($subset as $node) {
                if ($mode !== 0) {
                    $measure = $isModularity ? $strengths[$node] : $sizes[$node];
                    $threshold = $factor * $measure * ($subsetMeasure - $measure);
                } else {
                    $threshold = $this->objective->connectivityThreshold(
                        $this->objective->measure($strengths[$node], $sizes[$node]),
                        $subsetMeasure,
                        $totalEndpointWeight,
                    );
                }

                if ($external[$node] >= $threshold) {
                    $candidates[] = $node;
                }
            }

            /** @var list<int> $candidates */
            $candidates = $randomizer->shuffleArray($candidates);

            foreach ($candidates as $node) {
                $own = $membership[$node];

                // Merging is one-way: once a node has company it stays put.
                if ($memberCount[$own] !== 1) {
                    continue;
                }

                $end = $offsets[$node + 1];

                /** @var array<int, float> $toCommunity */
                $toCommunity = [];

                for ($i = $offsets[$node]; $i < $end; $i++) {
                    $neighbour = $targets[$i];

                    if ($neighbour === $node || ! isset($inside[$neighbour])) {
                        continue;
                    }

                    $community = $membership[$neighbour];
                    $toCommunity[$community] = ($toCommunity[$community] ?? 0.0) + $weights[$i];
                }

                $communityStrength[$own] = 0.0;
                $communitySize[$own] = 0.0;
                $memberCount[$own] = 0;

                $choices = [];
                $gains = [];
                $links = [];

                foreach ($toCommunity as $community => $weight) {
                    if ($community === $own) {
                        continue;
                    }

                    if ($mode !== 0) {
                        // Same inlining as moveNodes: the community's measure
                        // feeds both the connectivity threshold and the gain.
                        $measure = $isModularity ? $communityStrength[$community] : $communitySize[$community];

                        if ($external[$community] < $factor * $measure * ($subsetMeasure - $measure)) {
                            continue;
                        }

                        $gain = $weight
                            - $factor * ($isModularity ? $strengths[$node] : $sizes[$node]) * $measure;
                    } else {
                        $threshold = $this->objective->connectivityThreshold(
                            $this->objective->measure($communityStrength[$community], $communitySize[$community]),
                            $subsetMeasure,
                            $totalEndpointWeight,
                        );

                        if ($external[$community] < $threshold) {
                            continue;
                        }

                        $gain = $this->objective->gain(
                            $weight,
                            $strengths[$node],
                            $sizes[$node],
                            $communityStrength[$community],
                            $communitySize[$community],
                            $totalEndpointWeight,
                        );
                    }

                    if ($gain <= 0.0) {
                        continue;
                    }

                    $choices[] = $community;
                    $gains[] = $gain;
                    $links[] = $weight;
                }

                if ($gains === []) {
                    $communityStrength[$own] = $strengths[$node];
                    $communitySize[$own] = $sizes[$node];
                    $memberCount[$own] = 1;

                    continue;
                }

                $pick = $this->sample($gains, $randomizer);
                $chosen = $choices[$pick];

                $membership[$node] = $chosen;
                $communityStrength[$chosen] += $strengths[$node];
                $communitySize[$chosen] += $sizes[$node];
                $memberCount[$chosen]++;

                // Edges between the node and its new community were external to
                // both and are now internal to neither's boundary.
                $external[$chosen] += $external[$node] - 2.0 * $links[$pick];
                $external[$node] = 0.0;
            }
        }

        /** @var list<int> $membership */
        return Partition::fromMembership($membership);
    }

    /**
     * Phase 3. Each refined community becomes one node; edges between them are
     * summed, and edges inside one become its self-loop. Node strengths are
     * preserved exactly, so the objective is unchanged by aggregation and the
     * next pass optimises the same thing on a smaller graph.
     *
     * @param  list<float> $sizes
     * @return array{Graph, list<float>, list<int>}
     */
    private function aggregate(Graph $graph, Partition $refined, Partition $outer, array $sizes): array
    {
        $order = $refined->count();
        $membership = $refined->membership();
        $graphOrder = $graph->order();
        $directed = $graph->isDirected();
        [$offsets, $targets, $weights] = $graph->csr();

        // Accumulated on a flat from * order + to key, exactly as Graph does
        // internally, and walked over the raw CSR arrays: the edges()
        // generator would cost a resume and a fresh 3-element array per edge,
        // paid once per iteration of the whole algorithm.
        /** @var array<int, float> $accumulator */
        $accumulator = [];

        for ($from = 0; $from < $graphOrder; $from++) {
            $a = $membership[$from];
            $end = $offsets[$from + 1];

            for ($i = $offsets[$from]; $i < $end; $i++) {
                $to = $targets[$i];

                // Undirected storage holds both directions; count each once.
                if (! $directed && $to < $from) {
                    continue;
                }

                $b = $membership[$to];

                $key = $a <= $b ? $a * $order + $b : $b * $order + $a;

                if (isset($accumulator[$key])) {
                    $accumulator[$key] += $weights[$i];
                } else {
                    $accumulator[$key] = $weights[$i];
                }
            }
        }

        $edges = [];

        foreach ($accumulator as $key => $weight) {
            $edges[] = [intdiv($key, $order), $key % $order, $weight];
        }

        $aggregatedSizes = array_fill(0, $order, 0.0);
        $induced = [];
        $outerMembership = $outer->membership();

        foreach ($refined->communities() as $community => $members) {
            foreach ($members as $node) {
                $aggregatedSizes[$community] += $sizes[$node];
            }

            // A refined community lies wholly inside one phase-1 community, so
            // any member identifies which one.
            $induced[] = $outerMembership[$members[0]];
        }

        /** @var list<float> $aggregatedSizes */
        return [Graph::undirected($order, $edges), $aggregatedSizes, $induced];
    }

    /**
     * Choose among improving merges with probability rising in the gain.
     *
     * Always taking the best merge is what makes Louvain get stuck: the first
     * good split found is the only one ever tried. Sampling instead means a
     * slightly worse merge is sometimes taken, and the partition space gets
     * explored -- for free, because a worse merge is still an improvement.
     *
     * Weights are exponential in the gain, shifted by the maximum before
     * exponentiating so that a large gain cannot overflow, and scaled by that
     * maximum so `randomness` is dimensionless.
     *
     * @param  non-empty-list<float> $gains all strictly positive
     * @return int         index into the caller's parallel arrays
     */
    private function sample(array $gains, Randomizer $randomizer): int
    {
        $count = count($gains);

        if ($count === 1) {
            return 0;
        }

        $best = max($gains);
        $scale = $this->randomness * max(abs($best), PHP_FLOAT_EPSILON);

        $cumulative = [];
        $total = 0.0;

        foreach ($gains as $gain) {
            $total += exp(($gain - $best) / $scale);
            $cumulative[] = $total;
        }

        $target = $randomizer->getFloat(0.0, $total);

        foreach ($cumulative as $index => $ceiling) {
            if ($target <= $ceiling) {
                return $index;
            }
        }

        return $count - 1;
    }

    /**
     * @param list<int>   $vacated
     * @param list<float> $communitySize
     */
    private function takeVacatedSlot(array &$vacated, array $communitySize): ?int
    {
        while ($vacated !== []) {
            $slot = array_pop($vacated);

            if ($communitySize[$slot] <= 0.0) {
                return $slot;
            }
        }

        return null;
    }
}
