<?php

declare(strict_types=1);

namespace Vegoia\Graph;

use function array_fill;
use function array_key_exists;
use function count;

use Generator;

use function intdiv;
use function is_array;
use function is_finite;
use function ksort;

use Vegoia\Exception\InvalidArgument;

/**
 * An immutable graph stored in compressed sparse row form.
 *
 * Three parallel arrays hold the whole structure: `offsets` marks where each
 * node's neighbours begin, `targets` lists them contiguously, and `weights`
 * runs alongside. Reading a node's neighbourhood is then a sequential walk
 * over a packed array rather than a chase through per-edge objects.
 *
 * That choice is why this library has no collection dependency. A graph of
 * 100k edges is 200k integers and 200k floats here; the same graph expressed
 * as one object per edge costs both an allocation and a pointer dereference
 * per step, and community detection steps over every edge many times per
 * iteration. The abstraction would be paid for in the inner loop.
 *
 * Conventions, chosen to match igraph so the golden fixtures stay meaningful:
 *
 *   * parallel edges are merged, summing their weights;
 *   * an undirected self-loop contributes 2 to degree and 2w to strength;
 *   * `totalWeight()` sums each edge once, `totalEndpointWeight()` is the
 *     degree sum (2m), the quantity modularity is defined against.
 */
final class Graph
{
    /**
     * @param list<int>          $offsets   order + 1 entries
     * @param list<int>          $targets   neighbour list, grouped by source
     * @param list<float>        $weights   parallel to $targets
     * @param list<int>          $degrees
     * @param list<float>        $strengths
     * @param array<int, float>  $selfLoops node => loop weight, absent when zero
     */
    private function __construct(
        private readonly int $order,
        private readonly int $size,
        private readonly bool $directed,
        private readonly bool $weighted,
        private readonly array $offsets,
        private readonly array $targets,
        private readonly array $weights,
        private readonly array $degrees,
        private readonly array $strengths,
        private readonly array $selfLoops,
        private readonly float $totalWeight,
    ) {
    }

    /**
     * @param iterable<array{0: int, 1: int, 2?: float|int}> $edges
     */
    public static function undirected(int $order, iterable $edges = []): self
    {
        return self::build($order, $edges, directed: false);
    }

    /**
     * @param iterable<array{0: int, 1: int, 2?: float|int}> $edges
     */
    public static function directed(int $order, iterable $edges = []): self
    {
        return self::build($order, $edges, directed: true);
    }

    public function order(): int
    {
        return $this->order;
    }

    /** Distinct edges, counting a self-loop once. */
    public function size(): int
    {
        return $this->size;
    }

    public function isDirected(): bool
    {
        return $this->directed;
    }

    /**
     * Whether any stored edge carries a weight other than 1.
     *
     * Describes the graph, not the input it came from. Those differ: two
     * parallel edges of weight 1 merge into a single edge of weight 2, so a
     * construction with no explicit weights can still produce a weighted
     * graph. Reporting the input instead gave two identical graphs different
     * answers.
     */
    public function isWeighted(): bool
    {
        return $this->weighted;
    }

    public function isEmpty(): bool
    {
        return $this->order === 0;
    }

    /** @return Generator<int, int> */
    public function nodes(): Generator
    {
        for ($node = 0; $node < $this->order; $node++) {
            yield $node;
        }
    }

    public function degree(int $node): int
    {
        return $this->degrees[$this->assertNode($node)];
    }

    /** Weighted degree: the sum of incident edge weights. */
    public function strength(int $node): float
    {
        return $this->strengths[$this->assertNode($node)];
    }

    public function selfLoopWeight(int $node): float
    {
        return $this->selfLoops[$this->assertNode($node)] ?? 0.0;
    }

    /** @return Generator<int, float> neighbour => weight */
    public function neighbours(int $node): Generator
    {
        $this->assertNode($node);

        $end = $this->offsets[$node + 1];

        for ($i = $this->offsets[$node]; $i < $end; $i++) {
            yield $this->targets[$i] => $this->weights[$i];
        }
    }

    public function hasEdge(int $from, int $to): bool
    {
        return $this->locate($from, $to) !== null;
    }

    /** Zero when the edge is absent, which keeps callers free of null checks. */
    public function edgeWeight(int $from, int $to): float
    {
        $at = $this->locate($from, $to);

        return $at === null ? 0.0 : $this->weights[$at];
    }

    /**
     * Every edge once, in ascending order. For an undirected graph the pair is
     * emitted with the lower endpoint first.
     *
     * @return Generator<int, array{int, int, float}>
     */
    public function edges(): Generator
    {
        for ($node = 0; $node < $this->order; $node++) {
            $end = $this->offsets[$node + 1];

            for ($i = $this->offsets[$node]; $i < $end; $i++) {
                $target = $this->targets[$i];

                if (! $this->directed && $target < $node) {
                    continue;
                }

                yield [$node, $target, $this->weights[$i]];
            }
        }
    }

    /** Sum of edge weights, each edge counted once. */
    public function totalWeight(): float
    {
        return $this->totalWeight;
    }

    /** The degree sum, 2m for an undirected graph: modularity's normaliser. */
    public function totalEndpointWeight(): float
    {
        return $this->directed ? $this->totalWeight : 2.0 * $this->totalWeight;
    }

    /**
     * Raw CSR access for algorithm kernels.
     *
     * Community detection and Brandes' betweenness sweep the neighbour list
     * millions of times; a Generator allocates and resumes on every step,
     * which shows up plainly in a profile. These three accessors let a hot
     * loop index the packed arrays directly. Prefer `neighbours()` everywhere
     * that is not a measured bottleneck.
     *
     * @return array{list<int>, list<int>, list<float>} offsets, targets, weights
     */
    public function csr(): array
    {
        return [$this->offsets, $this->targets, $this->weights];
    }

    /** @return list<float> */
    public function strengths(): array
    {
        return $this->strengths;
    }

    /**
     * @param iterable<array{0: int, 1: int, 2?: float|int}> $edges
     */
    private static function build(int $order, iterable $edges, bool $directed): self
    {
        if ($order < 0) {
            throw InvalidArgument::negativeOrder($order);
        }

        // One flat map keyed by from * order + to instead of a hash per node.
        // The key packs the pair into a single integer, so merging costs one
        // hash probe rather than two, and a single ksort at the end delivers
        // every node's neighbour list already in ascending order -- see
        // compress() for why that falls out of the key layout.
        /** @var array<int, float> $merged */
        $merged = [];
        $selfLoops = [];
        $totalWeight = 0.0;

        foreach ($edges as $edge) {
            if (! is_array($edge) || ! array_key_exists(0, $edge) || ! array_key_exists(1, $edge)) {
                throw InvalidArgument::malformedEdge('expected [from, to] or [from, to, weight]');
            }

            $from = $edge[0];
            $to = $edge[1];
            $weight = (float) ($edge[2] ?? 1.0);

            if ($from < 0 || $from >= $order) {
                throw InvalidArgument::nodeOutOfRange($from, $order);
            }

            if ($to < 0 || $to >= $order) {
                throw InvalidArgument::nodeOutOfRange($to, $order);
            }

            if (! is_finite($weight)) {
                throw InvalidArgument::malformedEdge("edge {$from}-{$to} has a non-finite weight");
            }

            // Undirected edges are keyed by the ordered pair so that a-b and
            // b-a merge rather than becoming two edges.
            [$low, $high] = ! $directed && $from > $to ? [$to, $from] : [$from, $to];

            $key = $low * $order + $high;

            if (isset($merged[$key])) {
                $merged[$key] += $weight;
            } else {
                $merged[$key] = $weight;
            }

            $totalWeight += $weight;

            if ($low === $high) {
                $selfLoops[$low] = ($selfLoops[$low] ?? 0.0) + $weight;
            }
        }

        // Decided from the merged weights, after parallel edges have combined.
        $weighted = false;

        foreach ($merged as $weight) {
            if ($weight !== 1.0) {
                $weighted = true;

                break;
            }
        }

        return self::compress($order, $directed, $weighted, $merged, $selfLoops, $totalWeight);
    }

    /**
     * Counting-sort CSR assembly.
     *
     * Two passes over the merged edges: the first counts how many slots each
     * node needs, the second drops every entry straight into its final
     * position. Nothing is expanded into per-node hashes and nothing is sorted
     * per node -- the single ksort on the packed keys is enough, because keys
     * order by (from, to), so node u receives its neighbours v < u while the
     * edges (v, u) are processed (ascending in v), its self-loop next, and its
     * neighbours w > u during its own run of keys (ascending in w). Each list
     * comes out sorted by construction, which locate()'s binary search relies
     * on.
     *
     * @param array<int, float> $merged key = from * order + to
     * @param array<int, float> $selfLoops
     */
    private static function compress(
        int $order,
        bool $directed,
        bool $weighted,
        array $merged,
        array $selfLoops,
        float $totalWeight,
    ): self {
        $size = count($merged);
        ksort($merged);

        $slots = array_fill(0, $order, 0);

        foreach ($merged as $key => $weight) {
            $from = intdiv($key, $order);
            $to = $key % $order;
            $slots[$from]++;

            if (! $directed && $to !== $from) {
                $slots[$to]++;
            }
        }

        $offsets = [0];
        $cursor = [];
        $running = 0;

        for ($node = 0; $node < $order; $node++) {
            $cursor[] = $running;
            $running += $slots[$node];
            $offsets[] = $running;
        }

        $targets = array_fill(0, $running, 0);
        $weights = array_fill(0, $running, 0.0);
        $degrees = array_fill(0, $order, 0);
        $strengths = array_fill(0, $order, 0.0);

        foreach ($merged as $key => $weight) {
            $from = intdiv($key, $order);
            $to = $key % $order;

            $at = $cursor[$from]++;
            $targets[$at] = $to;
            $weights[$at] = $weight;

            if ($directed || $to === $from) {
                // A directed edge lives on one side only; an undirected
                // self-loop is stored once but joins the node to itself twice,
                // and both degree and strength must say so.
                $multiplicity = ! $directed && $to === $from ? 2 : 1;
                $degrees[$from] += $multiplicity;
                $strengths[$from] += $multiplicity * $weight;

                continue;
            }

            $at = $cursor[$to]++;
            $targets[$at] = $from;
            $weights[$at] = $weight;

            $degrees[$from]++;
            $degrees[$to]++;
            $strengths[$from] += $weight;
            $strengths[$to] += $weight;
        }

        /**
         * All four are array_fill over a contiguous range with only indexed
         * overwrites, so the keys stay 0..k-1.
         *
         * @var list<int>   $targets
         * @var list<float> $weights
         * @var list<int>   $degrees
         * @var list<float> $strengths
         */
        return new self(
            $order,
            $size,
            $directed,
            $weighted,
            $offsets,
            $targets,
            $weights,
            $degrees,
            $strengths,
            $selfLoops,
            $totalWeight,
        );
    }

    private function locate(int $from, int $to): ?int
    {
        $this->assertNode($from);
        $this->assertNode($to);

        // Neighbours are stored ascending, so a binary search beats a scan on
        // the high-degree nodes that dominate real graphs.
        $low = $this->offsets[$from];
        $high = $this->offsets[$from + 1] - 1;

        while ($low <= $high) {
            $mid = $low + (($high - $low) >> 1);
            $candidate = $this->targets[$mid];

            if ($candidate === $to) {
                return $mid;
            }

            if ($candidate < $to) {
                $low = $mid + 1;
            } else {
                $high = $mid - 1;
            }
        }

        return null;
    }

    private function assertNode(int $node): int
    {
        if ($node < 0 || $node >= $this->order) {
            throw InvalidArgument::nodeOutOfRange($node, $this->order);
        }

        return $node;
    }
}
