<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit\Graph\Community;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;
use Vegoia\Graph\Community\Quality\ConstantPotts;
use Vegoia\Graph\Community\Quality\Modularity;
use Vegoia\Graph\Graph;
use Vegoia\Graph\Partition;
use Vegoia\Tests\Support\GraphFixture;
use Vegoia\Tests\Support\Lre;

#[CoversClass(Modularity::class)]
#[CoversClass(ConstantPotts::class)]
final class QualityTest extends TestCase
{
    /**
     * Two nodes, one edge. Splitting them gives -1/2 and joining them gives 0.
     * Small enough to verify by hand from the definition, which anchors every
     * other modularity assertion in the suite.
     */
    public function test_modularity_of_the_smallest_interesting_graph(): void
    {
        $graph = Graph::undirected(2, [[0, 1]]);

        self::assertSame(-0.5, (new Modularity())->of($graph, Partition::singletons(2)));
        self::assertSame(0.0, (new Modularity())->of($graph, Partition::single(2)));
    }

    public function test_modularity_of_a_graph_with_no_edges_is_zero(): void
    {
        $graph = Graph::undirected(3);

        self::assertSame(0.0, (new Modularity())->of($graph, Partition::singletons(3)));
        self::assertSame(0.0, (new Modularity())->of($graph, Partition::single(3)));
    }

    public function test_constant_potts_counts_internal_weight_against_possible_pairs(): void
    {
        // A triangle: 3 internal edges, 3 possible pairs.
        $graph = Graph::undirected(3, [[0, 1], [1, 2], [0, 2]]);

        // resolution 1: 3 - 1 * 3 = 0
        self::assertSame(0.0, (new ConstantPotts(1.0))->of($graph, Partition::single(3)));
        // resolution 0.5: 3 - 0.5 * 3 = 1.5
        self::assertSame(1.5, (new ConstantPotts(0.5))->of($graph, Partition::single(3)));
        // singletons have no internal weight and no pairs
        self::assertSame(0.0, (new ConstantPotts(1.0))->of($graph, Partition::singletons(3)));
    }

    /** @return iterable<string, array{string, string, string}> */
    public static function probes(): iterable
    {
        foreach (GraphFixture::names() as $name) {
            foreach (['singletons', 'single_community', 'leiden_seed42'] as $probe) {
                foreach (['0.5', '1.0', '2.0'] as $resolution) {
                    yield "{$name}/{$probe}/y={$resolution}" => [$name, $probe, $resolution];
                }
            }
        }
    }

    #[DataProvider('probes')]
    public function test_modularity_agrees_with_igraph_exactly(string $name, string $probe, string $resolution): void
    {
        $fixture = GraphFixture::load($name);
        [$partition, $expected] = self::probe($fixture, $probe, 'modularity', $resolution);

        Lre::assertDigits(
            (new Modularity((float) $resolution))->of($fixture->graph(), $partition),
            $expected,
            "{$name}/{$probe}: modularity at resolution {$resolution}",
            digits: 12,
        );
    }

    /** @return iterable<string, array{string, string, string}> */
    public static function cpmProbes(): iterable
    {
        foreach (GraphFixture::names() as $name) {
            foreach (['singletons', 'single_community', 'leiden_seed42'] as $probe) {
                foreach (['0.05', '0.5', '1.0'] as $resolution) {
                    yield "{$name}/{$probe}/y={$resolution}" => [$name, $probe, $resolution];
                }
            }
        }
    }

    #[DataProvider('cpmProbes')]
    public function test_constant_potts_agrees_with_the_reference_exactly(string $name, string $probe, string $resolution): void
    {
        $fixture = GraphFixture::load($name);
        [$partition, $expected] = self::probe($fixture, $probe, 'cpm', $resolution);

        Lre::assertDigits(
            (new ConstantPotts((float) $resolution))->of($fixture->graph(), $partition),
            $expected,
            "{$name}/{$probe}: CPM at resolution {$resolution}",
            digits: 12,
        );
    }

    /** @return array{Partition, float} */
    private static function probe(GraphFixture $fixture, string $probe, string $measure, string $resolution): array
    {
        /** @var array{quality_probes: array<string, array{membership: list<int>, modularity: array<string, float>, cpm: array<string, float>}>} $expected */
        $expected = $fixture->expected;
        $entry = $expected['quality_probes'][$probe];

        return [Partition::fromMembership($entry['membership']), $entry[$measure][$resolution]];
    }
}
