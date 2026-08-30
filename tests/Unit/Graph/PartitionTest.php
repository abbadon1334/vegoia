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
}
