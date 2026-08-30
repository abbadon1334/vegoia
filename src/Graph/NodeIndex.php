<?php

declare(strict_types=1);

namespace Vegoia\Graph;

use function array_key_exists;
use function count;

use Vegoia\Exception\InvalidArgument;

/**
 * A two-way map between your identifiers and the integers the kernel uses.
 *
 * Graph stores nodes as a contiguous range 0..n-1 because that is what makes
 * compressed sparse row layout possible at all: with arbitrary keys, every
 * neighbour lookup becomes a hash lookup, and the packed arrays that make the
 * algorithms fast stop being arrays. Real data does not arrive that way -- it
 * arrives as UUIDs, slugs, or database ids -- so the translation lives here,
 * once, instead of being open-coded at every call site.
 *
 * Identifiers are numbered in order of first appearance, so building an index
 * from the same input twice gives the same numbering, and a partition computed
 * today can be compared with one computed last week.
 */
final class NodeIndex
{
    /** @var array<string, int> */
    private array $toInteger = [];

    /** @var list<string> */
    private array $toIdentifier = [];

    /** @param iterable<string> $identifiers */
    public static function of(iterable $identifiers = []): self
    {
        $index = new self();

        foreach ($identifiers as $identifier) {
            $index->add($identifier);
        }

        return $index;
    }

    /** Idempotent: adding a known identifier returns the number it already has. */
    public function add(string $identifier): int
    {
        if (array_key_exists($identifier, $this->toInteger)) {
            return $this->toInteger[$identifier];
        }

        $node = count($this->toIdentifier);
        $this->toInteger[$identifier] = $node;
        $this->toIdentifier[] = $identifier;

        return $node;
    }

    public function has(string $identifier): bool
    {
        return array_key_exists($identifier, $this->toInteger);
    }

    public function nodeFor(string $identifier): int
    {
        return $this->toInteger[$identifier]
            ?? throw InvalidArgument::malformedEdge("Unknown node identifier '{$identifier}'");
    }

    public function identifierFor(int $node): string
    {
        return $this->toIdentifier[$node]
            ?? throw InvalidArgument::nodeOutOfRange($node, count($this->toIdentifier));
    }

    public function count(): int
    {
        return count($this->toIdentifier);
    }

    /** @return list<string> */
    public function identifiers(): array
    {
        return $this->toIdentifier;
    }

    /**
     * Translate a partition back into your identifiers.
     *
     * @return array<string, int> identifier => community
     */
    public function label(Partition $partition): array
    {
        $labelled = [];

        foreach ($partition->membership() as $node => $community) {
            $labelled[$this->identifierFor($node)] = $community;
        }

        return $labelled;
    }

    /**
     * Build a graph from edges given as identifier pairs, registering any node
     * not seen before.
     *
     * @param  iterable<array{0: string, 1: string, 2?: float|int}> $edges
     * @param  iterable<string>                                     $nodes declare these to keep isolated nodes
     * @return array{Graph, self}
     */
    public static function graphFrom(iterable $edges, iterable $nodes = []): array
    {
        $index = self::of($nodes);
        $translated = [];

        foreach ($edges as $edge) {
            $translated[] = [
                $index->add($edge[0]),
                $index->add($edge[1]),
                (float) ($edge[2] ?? 1.0),
            ];
        }

        return [Graph::undirected($index->count(), $translated), $index];
    }
}
