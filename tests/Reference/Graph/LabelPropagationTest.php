<?php

declare(strict_types=1);

namespace Vegoia\Tests\Reference\Graph;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Vegoia\Graph\Community\Agreement;
use Vegoia\Graph\Community\LabelPropagation;
use Vegoia\Graph\Community\Leiden;
use Vegoia\Graph\Community\Quality\Modularity;
use Vegoia\Graph\Graph;
use Vegoia\Tests\Support\GraphFixture;
use Vegoia\Tests\Support\Paths;

/**
 * Label propagation, against the spread NetworkX produces.
 *
 * A spread and not an answer, because label propagation optimises nothing.
 * There is no quantity to say one run was better than another, no run is
 * reproducible across implementations, and on a graph without clear structure
 * the algorithm will collapse everything into one community or leave
 * everything apart -- NetworkX's own returns between one community and four on
 * the Petersen graph across fifty seeds. The width of the band is the honest
 * description of the algorithm, so what is asked here is whether this
 * implementation lands in the same territory.
 *
 * @see tools/generate_label_propagation_fixtures.py
 */
#[CoversClass(LabelPropagation::class)]
#[Group('reference')]
final class LabelPropagationTest extends TestCase
{
    private const int SEEDS = 50;

    /** @return iterable<string, array{string, array<string, mixed>}> */
    public static function graphs(): iterable
    {
        /** @var array{graphs: array<string, array<string, mixed>>} $decoded */
        $decoded = json_decode(
            (string) file_get_contents(Paths::fixture('label_propagation.json')),
            true,
            flags: JSON_THROW_ON_ERROR,
        );

        foreach ($decoded['graphs'] as $name => $entry) {
            yield $name => [$name, $entry];
        }
    }

    /**
     * @return array{non-empty-list<float>, non-empty-list<int>} modularity and
     *         community count, one of each per seed
     */
    private static function sweep(Graph $graph): array
    {
        $objective = new Modularity();
        $qualities = [];
        $counts = [];

        for ($seed = 0; $seed < self::SEEDS; $seed++) {
            $partition = new LabelPropagation($seed)->partition($graph);

            $qualities[] = $objective->of($graph, $partition);
            $counts[] = $partition->count();
        }

        /**
         * @var non-empty-list<float> $qualities
         * @var non-empty-list<int>   $counts
         */
        return [$qualities, $counts];
    }

    /**
     * The best of fifty runs must reach the best of the reference's fifty.
     *
     * The top of the band rather than the middle of it: with no objective
     * being maximised the mean says only how often the algorithm got lucky,
     * which depends on the order nodes happen to be visited in, while the best
     * says what the algorithm can find on this graph at all.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('graphs')]
    public function test_its_best_run_matches_the_references_best(string $name, array $entry): void
    {
        /** @var array{modularity: array{min: float, max: float}, communities: array{min: int, max: int}} $entry */
        [$qualities] = self::sweep(GraphFixture::load($name)->graph());

        self::assertGreaterThanOrEqual(
            $entry['modularity']['max'] - 1.0e-9,
            max($qualities),
            sprintf(
                '%s: best modularity over %d seeds was %.6f, the reference reaches %.6f',
                $name,
                self::SEEDS,
                max($qualities),
                $entry['modularity']['max'],
            ),
        );
    }

    /**
     * The number of communities must overlap the reference's range.
     *
     * Overlap rather than containment, for the reason the whole file exists:
     * two runs of an algorithm with no objective disagree, and so do two
     * implementations of it. Ranges that never meet would mean something else.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('graphs')]
    public function test_it_finds_as_many_communities_as_the_reference(string $name, array $entry): void
    {
        /** @var array{communities: array{min: int, max: int}} $entry */
        [, $counts] = self::sweep(GraphFixture::load($name)->graph());

        self::assertLessThanOrEqual($entry['communities']['max'], min($counts), $name);
        self::assertGreaterThanOrEqual($entry['communities']['min'], max($counts), $name);
    }

    /**
     * Where the answer is written on the face of the graph, it must be found
     * every time.
     *
     * Three cliques with nothing between them leave no room for the algorithm
     * to be unlucky: every node's neighbours are all in its own clique from
     * the first round onward. An implementation that got this wrong on any
     * seed would be broken rather than unstable, which is the distinction this
     * test exists to make.
     */
    public function test_it_always_finds_an_unambiguous_partition(): void
    {
        $graph = GraphFixture::load('disjoint_cliques')->graph();

        for ($seed = 0; $seed < self::SEEDS; $seed++) {
            $partition = new LabelPropagation($seed)->partition($graph);

            self::assertSame(3, $partition->count(), "seed {$seed}");
            self::assertSame(
                [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11]],
                $partition->communities(),
                "seed {$seed}",
            );
        }
    }

    /**
     * On a graph with obvious structure it must agree with Leiden.
     *
     * The two share no assumptions -- Leiden maximises a quality function,
     * this maximises nothing -- so agreement is evidence about the graph
     * rather than about either algorithm. On the ring of cliques, where the
     * answer is unambiguous, they must reach it together.
     */
    public function test_it_agrees_with_leiden_where_the_structure_is_obvious(): void
    {
        $graph = GraphFixture::load('ring_of_cliques_10x5')->graph();

        $best = null;
        $bestQuality = -INF;
        $objective = new Modularity();

        for ($seed = 0; $seed < self::SEEDS; $seed++) {
            $partition = new LabelPropagation($seed)->partition($graph);
            $quality = $objective->of($graph, $partition);

            if ($quality > $bestQuality) {
                $bestQuality = $quality;
                $best = $partition;
            }
        }

        self::assertNotNull($best);
        self::assertGreaterThan(
            0.95,
            Agreement::normalisedMutualInformation(
                Leiden::modularity(seed: 1)->partition($graph),
                $best,
            ),
            'the two algorithms disagree about a graph that is ten cliques in a ring',
        );
    }

    /** The same seed gives the same answer, or nothing here is reproducible. */
    public function test_a_seed_fixes_the_result(): void
    {
        $graph = GraphFixture::load('zachary')->graph();

        foreach ([0, 7, 42] as $seed) {
            self::assertSame(
                new LabelPropagation($seed)->partition($graph)->membership(),
                new LabelPropagation($seed)->partition($graph)->membership(),
                "seed {$seed}",
            );
        }
    }

    /**
     * Synchronous updating oscillates for ever on a bipartite graph, the two
     * halves swapping labels every round. This implementation is asynchronous
     * and must terminate; a complete bipartite graph is where that is tested,
     * not asserted.
     */
    public function test_it_terminates_on_a_bipartite_graph(): void
    {
        $edges = [];

        for ($left = 0; $left < 6; $left++) {
            for ($right = 6; $right < 12; $right++) {
                $edges[] = [$left, $right];
            }
        }

        $graph = Graph::undirected(12, $edges);

        for ($seed = 0; $seed < 10; $seed++) {
            $partition = new LabelPropagation($seed, maxIterations: 20)->partition($graph);

            self::assertSame(12, count($partition->membership()), "seed {$seed}");
        }
    }

    /**
     * Every node belongs to exactly one community, on every fixture.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('graphs')]
    public function test_it_returns_a_partition_of_every_node(string $name, array $entry): void
    {
        $graph = GraphFixture::load($name)->graph();
        $partition = new LabelPropagation(1)->partition($graph);

        $covered = 0;

        foreach ($partition->communities() as $members) {
            $covered += count($members);
        }

        self::assertSame($graph->order(), $covered, "{$name}: communities must tile the node set");
    }

    /** An isolated node keeps its own community: it has nothing to agree with. */
    public function test_an_isolated_node_is_its_own_community(): void
    {
        $graph = Graph::undirected(4, [[0, 1], [1, 2]]);
        $partition = new LabelPropagation(1)->partition($graph);

        self::assertCount(1, array_filter(
            $partition->communities(),
            static fn (array $members): bool => $members === [3],
        ));
    }

    public function test_an_empty_graph_gives_an_empty_partition(): void
    {
        self::assertSame(0, new LabelPropagation()->partition(Graph::undirected(0))->count());
    }
}
