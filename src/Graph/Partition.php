<?php

declare(strict_types=1);

namespace Vegoia\Graph;

use function array_fill;
use function array_values;
use function count;

use Vegoia\Exception\InvalidArgument;

/**
 * An assignment of every node to exactly one community.
 *
 * Community labels are canonicalised on construction: they are renumbered
 * 0..k-1 in order of first appearance. Without that, the same grouping could
 * be spelled a factorial number of ways, aggregation would leave gaps in the
 * numbering, and comparing two results would mean comparing set-of-sets by
 * hand at every call site.
 */
final readonly class Partition
{
    /**
     * @param list<int>       $membership node => community, already canonical
     * @param list<list<int>> $communities community => nodes, ascending
     */
    private function __construct(
        private array $membership,
        private array $communities,
    ) {
    }

    public static function singletons(int $order): self
    {
        if ($order < 0) {
            throw InvalidArgument::negativeOrder($order);
        }

        $membership = [];
        $communities = [];

        for ($node = 0; $node < $order; $node++) {
            $membership[] = $node;
            $communities[] = [$node];
        }

        return new self($membership, $communities);
    }

    /** Every node in one community. */
    public static function single(int $order): self
    {
        if ($order === 0) {
            return self::singletons(0);
        }

        return self::fromMembership(array_fill(0, $order, 0));
    }

    /** @param list<int> $membership */
    public static function fromMembership(array $membership): self
    {
        /** @var array<int, int> $canonical raw label => canonical label */
        $canonical = [];
        $normalised = [];
        $communities = [];

        foreach ($membership as $node => $label) {
            if (! isset($canonical[$label])) {
                $canonical[$label] = count($communities);
                $communities[] = [];
            }

            $community = $canonical[$label];
            $normalised[] = $community;
            $communities[$community][] = $node;
        }

        return new self($normalised, array_values($communities));
    }

    /** @return list<int> */
    public function membership(): array
    {
        return $this->membership;
    }

    /** @return list<list<int>> */
    public function communities(): array
    {
        return $this->communities;
    }

    public function count(): int
    {
        return count($this->communities);
    }

    public function order(): int
    {
        return count($this->membership);
    }

    public function communityOf(int $node): int
    {
        return $this->membership[$node]
            ?? throw InvalidArgument::nodeOutOfRange($node, count($this->membership));
    }

    /** @return list<int> */
    public function nodesIn(int $community): array
    {
        return $this->communities[$community]
            ?? throw InvalidArgument::outOfRange('Community index', (float) $community, 0.0, (float) (count($this->communities) - 1));
    }

    /** @return list<int> */
    public function sizes(): array
    {
        $sizes = [];

        foreach ($this->communities as $nodes) {
            $sizes[] = count($nodes);
        }

        return $sizes;
    }

    /** Same grouping, regardless of how the labels were spelled. */
    public function equals(self $other): bool
    {
        return $this->membership === $other->membership;
    }

    /**
     * Drop communities below a size, returning the nodes that were dropped.
     * Small communities are mostly noise in a retrieval pipeline, and the
     * caller usually needs to know which nodes became unassigned.
     *
     * @return array{Partition, list<int>} the kept partition and the orphans
     */
    public function withoutCommunitiesSmallerThan(int $minimumSize): array
    {
        $kept = [];
        $orphans = [];

        foreach ($this->communities as $nodes) {
            if (count($nodes) >= $minimumSize) {
                $kept[] = $nodes;
            } else {
                foreach ($nodes as $node) {
                    $orphans[] = $node;
                }
            }
        }

        $membership = $this->membership;

        foreach ($orphans as $node) {
            $membership[$node] = -1;
        }

        $rebuilt = [];
        $next = 0;

        foreach ($kept as $nodes) {
            foreach ($nodes as $node) {
                $membership[$node] = $next;
            }
            $rebuilt[] = $nodes;
            $next++;
        }

        // Orphans keep -1 so callers can distinguish "unassigned" from a real
        // community; membership() would otherwise silently relabel them into
        // community 0 on the next canonicalisation.
        /** @var list<int> $membership only values were reassigned */
        return [new self($membership, $rebuilt), $orphans];
    }
}
