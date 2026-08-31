<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit\Graph;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\TestCase;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Graph\NodeIndex;
use Vegoia\Graph\Partition;

#[CoversClass(NodeIndex::class)]
final class NodeIndexTest extends TestCase
{
    public function test_it_numbers_identifiers_in_order_of_first_appearance(): void
    {
        $index = NodeIndex::of(['alpha', 'beta', 'gamma']);

        self::assertSame(0, $index->nodeFor('alpha'));
        self::assertSame(2, $index->nodeFor('gamma'));
        self::assertSame('beta', $index->identifierFor(1));
        self::assertSame(3, $index->count());
    }

    public function test_adding_a_known_identifier_does_not_renumber_it(): void
    {
        $index = NodeIndex::of(['alpha']);

        self::assertSame(0, $index->add('alpha'));
        self::assertSame(1, $index->count());
    }

    public function test_it_reports_whether_it_knows_an_identifier(): void
    {
        $index = NodeIndex::of(['alpha', 'beta']);

        self::assertTrue($index->has('alpha'));
        self::assertFalse($index->has('gamma'));

        $index->add('gamma');

        self::assertTrue($index->has('gamma'));
    }

    public function test_it_translates_a_partition_back_into_identifiers(): void
    {
        $index = NodeIndex::of(['x', 'y', 'z']);

        self::assertSame(
            ['x' => 0, 'y' => 0, 'z' => 1],
            $index->label(Partition::fromMembership([0, 0, 1])),
        );
    }

    public function test_it_refuses_an_identifier_it_has_never_seen(): void
    {
        $this->expectException(InvalidArgument::class);

        NodeIndex::of(['a'])->nodeFor('b');
    }
}
