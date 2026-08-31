<?php

declare(strict_types=1);

namespace Vegoia\Interop;

use function array_key_exists;
use function count;
use function is_array;
use function is_int;
use function is_string;

use Vegoia\Exception\InvalidArgument;
use Vegoia\Graph\Graph;
use Vegoia\Graph\NodeIndex;
use Vegoia\Graph\Partition;

/**
 * A graph and the names its nodes came in with.
 *
 * The kernel numbers nodes 0..n-1 because that is what makes the compressed
 * sparse row layout possible at all, and real data never arrives that way --
 * it arrives as UUIDs, slugs, database ids or entity names. Translating in and
 * out is a small thing to get wrong repeatedly, so it happens once, here, and
 * the result carries both halves.
 *
 * The named constructors take the three shapes data actually turns up in.
 * None of them requires anything to be numbered, sorted, or complete: nodes
 * are discovered in order of first appearance, which makes the numbering
 * reproducible, so a partition computed today can be compared with one
 * computed last week from the same input.
 */
final readonly class LabelledGraph
{
    public function __construct(
        public Graph $graph,
        public NodeIndex $index,
    ) {
        if ($graph->order() !== $index->count()) {
            throw InvalidArgument::malformedEdge(
                "The graph has {$graph->order()} nodes and the index names {$index->count()}"
            );
        }
    }

    /**
     * From a list of edges, each a pair of names with an optional weight.
     *
     *     LabelledGraph::fromEdges([['alice', 'bob'], ['bob', 'carol', 2.5]]);
     *
     * `$nodes` names anything that must exist even though no edge mentions it.
     * Isolated nodes are not a degenerate case -- an entity nobody linked is
     * exactly what a knowledge graph should be able to report on.
     *
     * @param iterable<array{0: string|int, 1: string|int, 2?: float|int}> $edges
     * @param iterable<string|int>                                        $nodes
     */
    public static function fromEdges(iterable $edges, iterable $nodes = [], bool $directed = false): self
    {
        $index = new NodeIndex();

        foreach ($nodes as $node) {
            $index->add((string) $node);
        }

        $translated = [];

        foreach ($edges as $edge) {
            if (! is_array($edge) || ! array_key_exists(0, $edge) || ! array_key_exists(1, $edge)) {
                throw InvalidArgument::malformedEdge('expected [from, to] or [from, to, weight]');
            }

            $translated[] = [
                $index->add((string) $edge[0]),
                $index->add((string) $edge[1]),
                (float) ($edge[2] ?? 1.0),
            ];
        }

        return new self(
            $directed
                ? Graph::directed($index->count(), $translated)
                : Graph::undirected($index->count(), $translated),
            $index,
        );
    }

    /**
     * From a map of node to its neighbours.
     *
     *     LabelledGraph::fromAdjacency(['a' => ['b', 'c'], 'b' => ['c']]);
     *     LabelledGraph::fromAdjacency(['a' => ['b' => 2.5, 'c' => 1.0]]);
     *
     * Both shapes are accepted and told apart by whether the neighbour is the
     * value or the key, which is unambiguous: a list gives integer keys in
     * order, a weight map gives the neighbour's name as the key. A node with
     * no neighbours is written as an empty array and kept.
     *
     * @param iterable<string|int, iterable<string|int, string|int|float>> $adjacency
     */
    public static function fromAdjacency(iterable $adjacency, bool $directed = false): self
    {
        $index = new NodeIndex();
        $edges = [];

        foreach ($adjacency as $from => $neighbours) {
            $source = $index->add((string) $from);

            foreach ($neighbours as $key => $value) {
                // A weight map has the neighbour in the key; a plain list has
                // it in the value and an integer position in the key.
                $weighted = is_string($key);
                $neighbour = $weighted ? $key : $value;

                if (! is_string($neighbour) && ! is_int($neighbour)) {
                    throw InvalidArgument::malformedEdge(
                        "Neighbour of '{$from}' is neither a name nor a number"
                    );
                }

                $edges[] = [$source, $index->add((string) $neighbour), $weighted ? (float) $value : 1.0];
            }
        }

        return new self(
            $directed
                ? Graph::directed($index->count(), $edges)
                : Graph::undirected($index->count(), $edges),
            $index,
        );
    }

    /**
     * From a square adjacency matrix, with optional names for its rows.
     *
     * A zero means no edge; any other value is the weight. Without names the
     * rows are called "0", "1" and so on, so the identifiers still round-trip
     * and the result behaves like every other one here.
     *
     * @param list<list<float|int>> $matrix
     * @param list<string|int>      $labels
     */
    public static function fromMatrix(array $matrix, array $labels = [], bool $directed = false): self
    {
        $order = count($matrix);

        if ($labels !== [] && count($labels) !== $order) {
            throw InvalidArgument::malformedEdge(
                'The matrix has ' . $order . ' rows and ' . count($labels) . ' labels were given'
            );
        }

        $index = new NodeIndex();

        for ($row = 0; $row < $order; $row++) {
            $index->add((string) ($labels[$row] ?? $row));
        }

        $edges = [];

        foreach ($matrix as $row => $columns) {
            if (count($columns) !== $order) {
                throw InvalidArgument::malformedEdge(
                    "Row {$row} has " . count($columns) . " entries, not {$order}"
                );
            }

            foreach ($columns as $column => $weight) {
                if ((float) $weight === 0.0) {
                    continue;
                }

                // An undirected matrix states each edge twice; letting both
                // through would double every weight, since Graph sums parallel
                // edges. The upper triangle is enough, the diagonal included
                // so self-loops survive.
                if (! $directed && $column < $row) {
                    continue;
                }

                $edges[] = [$row, $column, (float) $weight];
            }
        }

        return new self(
            $directed ? Graph::directed($order, $edges) : Graph::undirected($order, $edges),
            $index,
        );
    }

    public function node(string|int $identifier): int
    {
        return $this->index->nodeFor((string) $identifier);
    }

    public function identifier(int $node): string
    {
        return $this->index->identifierFor($node);
    }

    /**
     * Which community each name landed in.
     *
     * @return array<string, int> name => community
     */
    public function membership(Partition $partition): array
    {
        return $this->index->label($partition);
    }

    /**
     * The communities themselves, each a list of names.
     *
     * The shape you want when reporting: "these entities belong together",
     * rather than a number per entity that has to be inverted first. Groups
     * come out in community order and names within a group in node order, so
     * the same input gives the same output.
     *
     * @return list<list<string>>
     */
    public function communities(Partition $partition): array
    {
        $groups = [];

        foreach ($partition->communities() as $members) {
            $names = [];

            foreach ($members as $node) {
                $names[] = $this->identifier($node);
            }

            $groups[] = $names;
        }

        return $groups;
    }

    /**
     * Any per-node result -- a centrality, a score, a distance -- keyed by name
     * instead of by position.
     *
     * @param list<float>|array<int, float> $byNode
     *
     * @return array<string, float>
     */
    public function named(array $byNode): array
    {
        $named = [];

        foreach ($byNode as $node => $value) {
            $named[$this->identifier($node)] = $value;
        }

        return $named;
    }
}
