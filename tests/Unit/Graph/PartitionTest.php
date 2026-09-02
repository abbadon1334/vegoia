<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit\Graph;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\TestCase;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Graph\Partition;

#[CoversClass(Partition::class)]
final class PartitionTest extends TestCase
{
    public function test_singletons_put_every_node_on_its_own(): void
    {
        $partition = Partition::singletons(3);

        self::assertSame(3, $partition->count());
        self::assertSame([[0], [1], [2]], $partition->communities());
    }

    /**
     * Labels arrive arbitrary -- an aggregation step leaves gaps, a caller may
     * pass anything. Normalising to 0..k-1 in order of first appearance makes
     * two equal partitions compare equal, which every downstream test relies on.
     */
    public function test_labels_are_normalised_in_order_of_first_appearance(): void
    {
        $partition = Partition::fromMembership([7, 7, 3, 9, 3]);

        self::assertSame([0, 0, 1, 2, 1], $partition->membership());
        self::assertSame(3, $partition->count());
        self::assertSame([[0, 1], [2, 4], [3]], $partition->communities());
    }

    public function test_it_reports_sizes_and_the_community_of_a_node(): void
    {
        $partition = Partition::fromMembership([0, 0, 1]);

        self::assertSame([2, 1], $partition->sizes());
        self::assertSame(0, $partition->communityOf(1));
        self::assertSame(1, $partition->communityOf(2));
    }

    public function test_it_lists_the_nodes_of_a_community(): void
    {
        $partition = Partition::fromMembership([0, 1, 0, 1, 2]);

        self::assertSame([0, 2], $partition->nodesIn(0));
        self::assertSame([4], $partition->nodesIn(2));

        $this->expectException(InvalidArgument::class);

        $partition->nodesIn(9);
    }

    public function test_two_partitions_are_equal_when_they_group_the_same_nodes(): void
    {
        self::assertTrue(
            Partition::fromMembership([5, 5, 2])->equals(Partition::fromMembership([0, 0, 1])),
            'the same grouping under different labels is the same partition',
        );

        self::assertFalse(
            Partition::fromMembership([0, 0, 1])->equals(Partition::fromMembership([0, 1, 1])),
        );
    }

    public function test_an_empty_partition_is_allowed(): void
    {
        self::assertSame(0, Partition::singletons(0)->count());
        self::assertSame([], Partition::fromMembership([])->communities());
    }

    public function test_it_rejects_a_node_outside_the_partition(): void
    {
        $this->expectException(InvalidArgument::class);

        Partition::fromMembership([0, 1])->communityOf(9);
    }

    /**
     * Filtering leaves the dropped nodes unassigned, and the result says so
     * rather than pretending they belong somewhere.
     *
     * The -1 is deliberate -- the source comment explains that relabelling
     * orphans into community 0 would be worse -- but it means the partition no
     * longer tiles the node set, and everything downstream has to know.
     */
    public function test_filtering_leaves_the_dropped_nodes_unassigned(): void
    {
        $partition = Partition::fromMembership([0, 0, 0, 1, 2, 2]);
        [$filtered, $orphans] = $partition->withoutCommunitiesSmallerThan(2);

        self::assertSame([3], $orphans);
        self::assertSame([0, 0, 0, -1, 1, 1], $filtered->membership());
        self::assertSame(2, $filtered->count());

        // The invariant everything else in this class keeps, and which a
        // filtered partition deliberately does not.
        self::assertSame(6, $filtered->order());
        self::assertSame(5, array_sum($filtered->sizes()));
        self::assertTrue($filtered->hasUnassigned());
        self::assertFalse($partition->hasUnassigned());
    }

    /**
     * An unassigned node is not a community of its own, and a round trip
     * through membership() would make it one.
     */
    public function test_a_filtered_partition_does_not_survive_a_round_trip(): void
    {
        [$filtered] = Partition::fromMembership([0, 0, 0, 1, 2, 2])
            ->withoutCommunitiesSmallerThan(2);

        self::assertSame(2, $filtered->count());
        self::assertSame(
            3,
            Partition::fromMembership($filtered->membership())->count(),
            'rebuilding turns the orphans into a third community, which is why hasUnassigned() '
            . 'exists and why the comparison measures refuse such a partition',
        );
    }
}
