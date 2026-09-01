<?php

declare(strict_types=1);

namespace Vegoia\Tests\Reference\Graph;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Graph\Connectivity;
use Vegoia\Graph\Graph;
use Vegoia\Tests\Support\GraphFixture;
use Vegoia\Tests\Support\Paths;

/**
 * Strong connectivity, bridges and articulation points, against NetworkX.
 *
 * The fixture sorts components within and between, and bridges as sorted pairs
 * sorted again, because neither order carries meaning -- pinning NetworkX's
 * traversal order would fail any implementation that walked the graph
 * differently, which is a test of the walk rather than of the answer.
 *
 * @see tools/generate_structure_fixtures.py
 */
#[CoversClass(Connectivity::class)]
#[Group('reference')]
final class StructureTest extends TestCase
{
    /** @var array<string, mixed>|null */
    private static ?array $fixture = null;

    /** @return array<string, mixed> */
    private static function section(string $name): array
    {
        if (self::$fixture === null) {
            /** @var array<string, mixed> $decoded */
            $decoded = json_decode(
                (string) file_get_contents(Paths::fixture('structure.json')),
                true,
                flags: JSON_THROW_ON_ERROR,
            );
            self::$fixture = $decoded;
        }

        /** @var array<string, mixed> $out */
        $out = self::$fixture[$name];

        return $out;
    }

    /** @return iterable<string, array{string, array<string, mixed>}> */
    public static function directedGraphs(): iterable
    {
        /** @var array<string, array<string, mixed>> $entries */
        $entries = self::section('directed');

        foreach ($entries as $name => $entry) {
            yield $name => [$name, $entry];
        }
    }

    /** @return iterable<string, array{string, array<string, mixed>}> */
    public static function undirectedGraphs(): iterable
    {
        /** @var array<string, array<string, mixed>> $entries */
        $entries = self::section('undirected');

        foreach ($entries as $name => $entry) {
            yield $name => [$name, $entry];
        }
    }

    /** @param array<string, mixed> $entry */
    #[DataProvider('directedGraphs')]
    public function test_the_strongly_connected_components_match(string $name, array $entry): void
    {
        /** @var array{components: list<list<int>>, is_strongly_connected: bool, is_directed_acyclic: bool} $entry */
        $graph = GraphFixture::directed($name)->directedGraph();

        $found = array_map(array_values(...), Connectivity::stronglyConnectedComponents($graph)->communities());
        usort($found, static fn (array $a, array $b): int => $a[0] <=> $b[0]);

        self::assertSame($entry['components'], $found, $name);
        self::assertSame($entry['is_strongly_connected'], Connectivity::isStronglyConnected($graph), $name);
        self::assertSame($entry['is_directed_acyclic'], Connectivity::isAcyclic($graph), $name);
    }

    /**
     * Strong components refine weak ones: two nodes that can reach each other
     * following the arrows can certainly reach each other ignoring them.
     *
     * Free to state and impossible to satisfy by accident. An implementation
     * that quietly ignored direction would return the weak components here and
     * fail on cycle_with_tail, whose three nodes in a cycle and two hanging
     * off it are one weak component and three strong ones.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('directedGraphs')]
    public function test_every_strong_component_sits_inside_one_weak_component(
        string $name,
        array $entry,
    ): void {
        /** @var array{weak_components: list<list<int>>} $entry */
        $graph = GraphFixture::directed($name)->directedGraph();

        $weak = [];

        foreach ($entry['weak_components'] as $index => $members) {
            foreach ($members as $member) {
                $weak[$member] = $index;
            }
        }

        foreach (Connectivity::stronglyConnectedComponents($graph)->communities() as $members) {
            $labels = [];

            foreach ($members as $member) {
                $labels[$weak[$member]] = true;
            }

            self::assertCount(
                1,
                $labels,
                "{$name}: a strongly connected component spans more than one weak component",
            );
        }
    }

    /** @param array<string, mixed> $entry */
    #[DataProvider('undirectedGraphs')]
    public function test_the_bridges_and_articulation_points_match(string $name, array $entry): void
    {
        /** @var array{bridges: list<array{int, int}>, articulation_points: list<int>} $entry */
        $graph = GraphFixture::load($name)->graph();

        self::assertSame($entry['bridges'], Connectivity::bridges($graph), "{$name}: bridges");
        self::assertSame(
            $entry['articulation_points'],
            Connectivity::articulationPoints($graph),
            "{$name}: articulation points",
        );
    }

    /**
     * The definition, checked directly: removing a bridge must raise the
     * component count, and removing any other edge must not.
     *
     * This is the assertion that would survive the fixture being wrong. It
     * rebuilds the graph without one edge and counts, which is a different
     * computation from the depth-first characterisation the implementation
     * uses, so agreeing means the characterisation was applied correctly
     * rather than that two copies of it agree.
     *
     * It runs on every fixture. It used to skip anything above a hundred
     * edges as "quadratic", which was a guess rather than a measurement: the
     * largest here is Les Misérables at 254 edges and it takes 106 ms.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('undirectedGraphs')]
    public function test_a_bridge_is_exactly_an_edge_whose_removal_disconnects(
        string $name,
        array $entry,
    ): void {
        $fixture = GraphFixture::load($name);
        $graph = $fixture->graph();

        $before = Connectivity::components($graph)->count();
        $bridges = array_map(
            static fn (array $pair): string => "{$pair[0]}-{$pair[1]}",
            Connectivity::bridges($graph),
        );

        foreach ($graph->edges() as [$from, $to]) {
            $remaining = [];

            foreach ($graph->edges() as [$u, $v, $w]) {
                if ($u !== $from || $v !== $to) {
                    $remaining[] = [$u, $v, $w];
                }
            }

            $without = Graph::undirected($graph->order(), $remaining);
            $disconnects = Connectivity::components($without)->count() > $before;

            self::assertSame(
                $disconnects,
                in_array("{$from}-{$to}", $bridges, strict: true),
                sprintf(
                    '%s: removing %d-%d %s the graph, and it is %s a bridge',
                    $name,
                    $from,
                    $to,
                    $disconnects ? 'disconnects' : 'does not disconnect',
                    in_array("{$from}-{$to}", $bridges, strict: true) ? 'listed as' : 'not listed as',
                ),
            );
        }
    }

    /**
     * Every bridge's endpoints are articulation points, unless the endpoint
     * has degree one -- a leaf cannot cut anything off by leaving, because
     * there is nothing beyond it.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('undirectedGraphs')]
    public function test_a_bridge_endpoint_is_a_cut_vertex_unless_it_is_a_leaf(
        string $name,
        array $entry,
    ): void {
        $graph = GraphFixture::load($name)->graph();
        $points = Connectivity::articulationPoints($graph);
        $bridges = Connectivity::bridges($graph);

        // Six of these fixtures are 2-edge-connected and have no bridges at
        // all, so the loop below would assert nothing. Saying so is the
        // difference between a test that found nothing and one that ran
        // nothing -- and it pins the fact, which is itself worth pinning:
        // a complete graph or a ring of cliques has no load-bearing edge.
        if ($bridges === []) {
            self::assertSame([], $bridges, "{$name} has no bridges, so there is nothing to imply");

            return;
        }

        foreach ($bridges as [$from, $to]) {
            foreach ([$from, $to] as $node) {
                if ($graph->degree($node) === 1) {
                    continue;
                }

                self::assertContains(
                    $node,
                    $points,
                    "{$name}: {$node} ends a bridge and has degree above one, so it must be a cut vertex",
                );
            }
        }
    }

    public function test_the_undirected_questions_refuse_a_directed_graph(): void
    {
        $graph = Graph::directed(3, [[0, 1], [1, 2]]);

        $this->expectException(InvalidArgument::class);
        $this->expectExceptionMessageMatches('/undirected graphs only/');

        Connectivity::bridges($graph);
    }

    public function test_articulation_points_refuse_a_directed_graph(): void
    {
        $graph = Graph::directed(3, [[0, 1], [1, 2]]);

        $this->expectException(InvalidArgument::class);
        $this->expectExceptionMessageMatches('/undirected graphs only/');

        Connectivity::articulationPoints($graph);
    }

    /** An empty graph has nothing to say, and must say it without failing. */
    public function test_an_empty_graph_is_answerable(): void
    {
        $empty = Graph::undirected(0);

        self::assertSame([], Connectivity::bridges($empty));
        self::assertSame([], Connectivity::articulationPoints($empty));
        self::assertSame(0, Connectivity::stronglyConnectedComponents($empty)->count());
        self::assertTrue(Connectivity::isStronglyConnected($empty));
        self::assertTrue(Connectivity::isAcyclic($empty));
    }

    /** A self-loop is a cycle, however short. */
    public function test_a_self_loop_makes_a_graph_cyclic(): void
    {
        self::assertTrue(Connectivity::isAcyclic(Graph::directed(2, [[0, 1]])));
        self::assertFalse(Connectivity::isAcyclic(Graph::directed(2, [[0, 1], [1, 1]])));
    }
}
