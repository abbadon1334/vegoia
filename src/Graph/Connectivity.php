<?php

declare(strict_types=1);

namespace Vegoia\Graph;

use function array_fill;
use function array_pop;
use function count;
use function min;
use function sort;

use Vegoia\Exception\InvalidArgument;

/**
 * Connectivity questions answered with an iterative breadth-first sweep.
 *
 * Iterative rather than recursive on purpose: a long path in a large graph
 * would otherwise be limited by the call stack, and PHP gives no useful
 * diagnostic when it runs out.
 *
 * Directed graphs split the question in two. Weak connectivity ignores the
 * arrows and asks whether the drawing is in one piece; strong connectivity
 * respects them and asks whether every node can reach every other. Only the
 * second is an equivalence relation, and only the second tells you whether a
 * citation graph has real cycles or merely looks like it.
 *
 * `inducesConnectedSubgraph()` is the reason this class exists at all. The
 * guarantee that distinguishes Leiden from Louvain is that every community it
 * returns is internally connected -- Louvain can and does return communities
 * split into pieces that are only joined through nodes outside them. That
 * property is checkable directly, with no golden values involved, so it is the
 * strongest assertion available about a stochastic algorithm.
 */
final class Connectivity
{
    public static function isConnected(Graph $graph): bool
    {
        return $graph->order() === 0 || self::components($graph)->count() === 1;
    }

    public static function components(Graph $graph): Partition
    {
        $order = $graph->order();

        if ($order === 0) {
            return Partition::fromMembership([]);
        }

        [$offsets, $targets] = $graph->csr();

        /** @var list<int> $component */
        $component = array_fill(0, $order, -1);
        $found = 0;

        for ($root = 0; $root < $order; $root++) {
            if ($component[$root] !== -1) {
                continue;
            }

            $component[$root] = $found;
            $frontier = [$root];

            while ($frontier !== []) {
                $node = array_pop($frontier);
                $end = $offsets[$node + 1];

                for ($i = $offsets[$node]; $i < $end; $i++) {
                    $neighbour = $targets[$i];

                    if ($component[$neighbour] === -1) {
                        $component[$neighbour] = $found;
                        $frontier[] = $neighbour;
                    }
                }
            }

            $found++;
        }

        return Partition::fromMembership($component);
    }

    /**
     * Is the subgraph induced by these nodes connected?
     *
     * Only edges with *both* endpoints inside the set count -- a path leaving
     * the set and coming back does not make it connected, which is exactly the
     * distinction Leiden's guarantee turns on.
     *
     * @param list<int> $nodes
     */
    public static function inducesConnectedSubgraph(Graph $graph, array $nodes): bool
    {
        $size = count($nodes);

        if ($size <= 1) {
            return true;
        }

        $inside = [];
        foreach ($nodes as $node) {
            $inside[$node] = true;
        }

        [$offsets, $targets] = $graph->csr();

        $seen = [$nodes[0] => true];
        $frontier = [$nodes[0]];
        $reached = 1;

        while ($frontier !== []) {
            $node = array_pop($frontier);
            $end = $offsets[$node + 1];

            for ($i = $offsets[$node]; $i < $end; $i++) {
                $neighbour = $targets[$i];

                if (! isset($inside[$neighbour]) || isset($seen[$neighbour])) {
                    continue;
                }

                $seen[$neighbour] = true;
                $frontier[] = $neighbour;
                $reached++;
            }
        }

        return $reached === $size;
    }

    /**
     * Strongly connected components: the groups that can all reach each other.
     *
     * Tarjan's algorithm, in one pass. Each node is given the order it was
     * first seen and a low-link -- the earliest node reachable from its
     * subtree by tree edges and at most one back edge. A node whose low-link
     * never improves on its own index is the root of a component, and
     * everything stacked above it belongs to that component.
     *
     * Iterative rather than recursive. Tarjan is usually written recursively
     * and the recursion is one frame per node: a chain of 100k nodes, which is
     * a perfectly ordinary shape for a citation or dependency graph, would
     * exhaust the stack and PHP would say nothing useful about why. The
     * explicit stack costs a state machine and buys a guarantee.
     *
     * On an undirected graph this is the same as components(); the arrows are
     * what make the two differ, so calling it there is allowed rather than
     * refused.
     */
    public static function stronglyConnectedComponents(Graph $graph): Partition
    {
        $order = $graph->order();

        if ($order === 0) {
            return Partition::fromMembership([]);
        }

        [$offsets, $targets] = $graph->csr();

        $index = array_fill(0, $order, -1);
        $lowLink = array_fill(0, $order, 0);
        $onStack = array_fill(0, $order, false);
        $membership = array_fill(0, $order, -1);

        $stack = [];
        $next = 0;
        $component = 0;

        for ($root = 0; $root < $order; $root++) {
            if ($index[$root] !== -1) {
                continue;
            }

            // Each frame is [node, the position in its neighbour run we have
            // reached]. Resuming a frame means continuing that run, which is
            // exactly what returning from the recursive call would do.
            $frames = [[$root, $offsets[$root]]];
            $index[$root] = $lowLink[$root] = $next++;
            $stack[] = $root;
            $onStack[$root] = true;

            while ($frames !== []) {
                [$node, $cursor] = $frames[count($frames) - 1];

                if ($cursor < $offsets[$node + 1]) {
                    $frames[count($frames) - 1][1] = $cursor + 1;
                    $neighbour = $targets[$cursor];

                    if ($index[$neighbour] === -1) {
                        $index[$neighbour] = $lowLink[$neighbour] = $next++;
                        $stack[] = $neighbour;
                        $onStack[$neighbour] = true;
                        $frames[] = [$neighbour, $offsets[$neighbour]];
                    } elseif ($onStack[$neighbour]) {
                        // A back edge to something still open. Only the index
                        // is taken, never the low-link: the neighbour may yet
                        // improve, and copying a value that is not final is
                        // how this algorithm is most often got wrong.
                        $lowLink[$node] = min($lowLink[$node], $index[$neighbour]);
                    }

                    continue;
                }

                array_pop($frames);

                if ($frames !== []) {
                    $parent = $frames[count($frames) - 1][0];
                    $lowLink[$parent] = min($lowLink[$parent], $lowLink[$node]);
                }

                if ($lowLink[$node] === $index[$node]) {
                    // The stack holds every node discovered since this one and
                    // not yet assigned, so it always contains at least $node
                    // itself and this cannot run dry.
                    do {
                        /** @var int $member */
                        $member = array_pop($stack);
                        $onStack[$member] = false;
                        $membership[$member] = $component;
                    } while ($member !== $node);

                    $component++;
                }
            }
        }

        /** @var list<int> $membership */
        return Partition::fromMembership($membership);
    }

    /** Whether every node can reach every other, following the arrows. */
    public static function isStronglyConnected(Graph $graph): bool
    {
        return $graph->order() === 0 || self::stronglyConnectedComponents($graph)->count() === 1;
    }

    /**
     * Whether a directed graph has no cycles.
     *
     * Read off the strongly connected components rather than by a separate
     * traversal: a cycle is precisely a component with more than one node, or
     * a self-loop. Two answers from one walk cannot disagree with each other.
     */
    public static function isAcyclic(Graph $graph): bool
    {
        foreach (self::stronglyConnectedComponents($graph)->communities() as $members) {
            if (count($members) > 1) {
                return false;
            }
        }

        for ($node = 0; $node < $graph->order(); $node++) {
            if ($graph->selfLoopWeight($node) !== 0.0) {
                return false;
            }
        }

        return true;
    }

    /**
     * The edges whose removal would disconnect the graph.
     *
     * Where a graph is fragile. For a knowledge graph assembled by extraction,
     * a bridge is a single stated relation carrying an entire region of the
     * answer -- if that extraction was wrong, everything beyond it becomes
     * unreachable and nothing else says so. It is the edge worth having a
     * human look at.
     *
     * @return list<array{int, int}> each pair with the lower endpoint first,
     *                               in ascending order
     */
    public static function bridges(Graph $graph): array
    {
        self::assertUndirected($graph, 'Finding bridges');

        return self::fragility($graph)[0];
    }

    /**
     * The nodes whose removal would disconnect the graph.
     *
     * The same question one level up: a bridge says a relation is
     * load-bearing, an articulation point says an entity is.
     *
     * @return list<int> in ascending order
     */
    public static function articulationPoints(Graph $graph): array
    {
        self::assertUndirected($graph, 'Finding articulation points');

        return self::fragility($graph)[1];
    }

    /**
     * Both fragility questions, from one depth-first walk.
     *
     * Hopcroft and Tarjan's characterisation. Give each node the order it was
     * discovered and a low-link -- the earliest node its subtree can reach by
     * tree edges and at most one edge back up. Then for a tree edge from u to
     * a child v:
     *
     *   * the edge is a bridge when low[v] > disc[u], meaning nothing under v
     *     reaches u or anything above it, so cutting the edge strands the
     *     subtree;
     *   * u is an articulation point when low[v] >= disc[u], the weaker
     *     condition, because v's subtree may reach u itself and still be cut
     *     off by removing u.
     *
     * The root is the exception, and the one everybody gets wrong: it has no
     * parent side to be separated from, so it is a cut vertex only when it has
     * more than one child in the tree.
     *
     * Iterative, like the rest of this class. Recursion here is one frame per
     * node, and a path of 100k nodes -- an ordinary shape for a chain of
     * references -- would exhaust the stack with no useful diagnostic.
     *
     * @return array{list<array{int, int}>, list<int>}
     */
    private static function fragility(Graph $graph): array
    {
        $order = $graph->order();

        if ($order === 0) {
            return [[], []];
        }

        [$offsets, $targets] = $graph->csr();

        $discovered = array_fill(0, $order, -1);
        $lowLink = array_fill(0, $order, 0);
        $isArticulation = array_fill(0, $order, false);

        $bridges = [];
        $next = 0;

        for ($root = 0; $root < $order; $root++) {
            if ($discovered[$root] !== -1) {
                continue;
            }

            $discovered[$root] = $lowLink[$root] = $next++;
            $rootChildren = 0;

            // [node, cursor into its neighbour run, the node it was reached
            // from]. The parent is carried so the edge back up can be skipped
            // once -- parallel edges are merged by the graph, so there is
            // exactly one and skipping by node identity is right.
            $frames = [[$root, $offsets[$root], -1]];

            while ($frames !== []) {
                $top = count($frames) - 1;
                [$node, $cursor, $parent] = $frames[$top];

                if ($cursor < $offsets[$node + 1]) {
                    $frames[$top][1] = $cursor + 1;
                    $neighbour = $targets[$cursor];

                    if ($neighbour === $parent || $neighbour === $node) {
                        continue;
                    }

                    if ($discovered[$neighbour] === -1) {
                        if ($node === $root) {
                            $rootChildren++;
                        }

                        $discovered[$neighbour] = $lowLink[$neighbour] = $next++;
                        $frames[] = [$neighbour, $offsets[$neighbour], $node];
                    } else {
                        $lowLink[$node] = min($lowLink[$node], $discovered[$neighbour]);
                    }

                    continue;
                }

                array_pop($frames);

                if ($parent < 0) {
                    continue;
                }

                // $parent is a real node here: the -1 that marks a root was
                // handled by the continue above.
                $lowLink[$parent] = min($lowLink[$parent], $lowLink[$node]);

                if ($lowLink[$node] > $discovered[$parent]) {
                    $bridges[] = [min($parent, $node), max($parent, $node)];
                }

                if ($parent !== $root && $lowLink[$node] >= $discovered[$parent]) {
                    $isArticulation[$parent] = true;
                }
            }

            if ($rootChildren > 1) {
                $isArticulation[$root] = true;
            }
        }

        sort($bridges);

        $points = [];

        foreach ($isArticulation as $node => $yes) {
            if ($yes) {
                $points[] = $node;
            }
        }

        return [$bridges, $points];
    }

    private static function assertUndirected(Graph $graph, string $operation): void
    {
        if ($graph->isDirected()) {
            throw InvalidArgument::directedNotSupported(
                $operation,
                'the directed analogue is a dominator computation on a different tree, not the '
                . 'same walk with the arrows ignored.',
            );
        }
    }
}
