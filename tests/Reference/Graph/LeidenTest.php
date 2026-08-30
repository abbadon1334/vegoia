<?php

declare(strict_types=1);

namespace Vegoia\Tests\Reference\Graph;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Vegoia\Graph\Community\Leiden;
use Vegoia\Graph\Community\Quality\Modularity;
use Vegoia\Graph\Connectivity;
use Vegoia\Graph\Graph;
use Vegoia\Tests\Support\GraphFixture;

/**
 * Leiden against the reference implementation.
 *
 * A stochastic algorithm cannot be pinned to one expected partition: our RNG
 * stream is not Python's, and even leidenalg gives different answers on
 * different seeds. So the suite asserts three separable things instead.
 *
 *   1. What is deterministic is checked exactly. On disjoint cliques there is
 *      one defensible answer and the algorithm must find it.
 *
 *   2. What is stochastic is checked against a measured envelope. The fixtures
 *      record the modularity leidenalg reaches over 50 seeds; landing inside
 *      that band means our search is as good as the reference's, while a
 *      broken refinement step drops well below it.
 *
 *   3. What the paper *guarantees* is checked as a property, on every graph
 *      and every seed. Leiden's headline result over Louvain is that no
 *      community is internally disconnected. That needs no golden value, and
 *      it is the assertion most likely to catch a subtly wrong implementation.
 *
 * @see V.A. Traag, L. Waltman & N.J. van Eck (2019), "From Louvain to Leiden:
 *      guaranteeing well-connected communities", Scientific Reports 9, 5233.
 */
#[CoversClass(Leiden::class)]
#[Group('reference')]
#[Group('leiden')]
final class LeidenTest extends TestCase
{
    /** @return iterable<string, array{string}> */
    public static function fixtures(): iterable
    {
        foreach (GraphFixture::names() as $name) {
            yield $name => [$name];
        }
    }

    #[DataProvider('fixtures')]
    public function test_every_community_it_returns_is_internally_connected(string $name): void
    {
        $fixture = GraphFixture::load($name);
        $graph = $fixture->graph();

        // Across many seeds: the guarantee is unconditional, not typical.
        for ($seed = 1; $seed <= 20; $seed++) {
            $partition = Leiden::modularity(seed: $seed)->partition($graph);

            foreach ($partition->communities() as $index => $members) {
                self::assertTrue(
                    Connectivity::inducesConnectedSubgraph($graph, $members),
                    "{$name}: seed {$seed} produced community {$index} that is internally disconnected, "
                    . 'which the Leiden guarantee forbids',
                );
            }
        }
    }

    #[DataProvider('fixtures')]
    public function test_it_returns_a_valid_partition_of_every_node(string $name): void
    {
        $fixture = GraphFixture::load($name);
        $partition = Leiden::modularity(seed: 42)->partition($fixture->graph());

        self::assertSame($fixture->nodes, $partition->order());
        self::assertCount($fixture->nodes, $partition->membership());

        $covered = 0;
        foreach ($partition->communities() as $members) {
            $covered += count($members);
        }

        self::assertSame($fixture->nodes, $covered, 'communities must tile the node set');
    }

    #[DataProvider('fixtures')]
    public function test_modularity_lands_inside_the_envelope_measured_from_leidenalg(string $name): void
    {
        $fixture = GraphFixture::load($name);
        $graph = $fixture->graph();
        $envelope = $fixture->leidenModularityEnvelope();
        $objective = new Modularity();

        $best = -INF;
        for ($seed = 1; $seed <= 10; $seed++) {
            $q = $objective->of($graph, Leiden::modularity(seed: $seed)->partition($graph));
            $best = max($best, $q);
        }

        // A different RNG explores in a different order, so require reaching
        // the bottom of the reference band rather than reproducing it.
        self::assertGreaterThanOrEqual(
            $envelope['min'] - 1.0e-9,
            $best,
            sprintf(
                "%s: best modularity over 10 seeds was %.9f, below leidenalg's worst of %.9f "
                . '(reference band [%.9f, %.9f])',
                $name,
                $best,
                $envelope['min'],
                $envelope['min'],
                $envelope['max'],
            ),
        );
    }

    /**
     * Three cliques with no edges between them. Every sensible objective agrees
     * on the answer, so this is an equality test, not an envelope test: if the
     * algorithm cannot find a partition that is written on the face of the
     * graph, nothing else it reports is worth reading.
     */
    public function test_it_finds_the_exact_partition_when_the_answer_is_unambiguous(): void
    {
        $fixture = GraphFixture::load('disjoint_cliques');
        $graph = $fixture->graph();

        for ($seed = 1; $seed <= 20; $seed++) {
            $partition = Leiden::modularity(seed: $seed)->partition($graph);

            self::assertSame(3, $partition->count(), "seed {$seed}");
            self::assertSame(
                [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11]],
                $partition->communities(),
                "seed {$seed}: the three cliques are the only defensible partition",
            );
        }
    }

    public function test_it_separates_a_ring_of_cliques_into_its_cliques(): void
    {
        $fixture = GraphFixture::load('ring_of_cliques_10x5');
        $graph = $fixture->graph();
        $partition = Leiden::modularity(seed: 7)->partition($graph);

        self::assertSame(10, $partition->count());

        foreach ($partition->communities() as $members) {
            self::assertCount(5, $members);
        }
    }

    public function test_the_same_seed_gives_the_same_partition(): void
    {
        $graph = GraphFixture::load('zachary')->graph();

        $first = Leiden::modularity(seed: 1234)->partition($graph);
        $second = Leiden::modularity(seed: 1234)->partition($graph);

        self::assertTrue(
            $first->equals($second),
            'a fixed seed must be reproducible, or a pipeline cannot be re-run',
        );
    }

    public function test_a_graph_with_no_edges_stays_in_singletons(): void
    {
        $partition = Leiden::modularity(seed: 1)->partition(Graph::undirected(5));

        self::assertSame(5, $partition->count());
    }

    public function test_an_empty_graph_yields_an_empty_partition(): void
    {
        self::assertSame(0, Leiden::modularity()->partition(Graph::undirected(0))->count());
    }

    public function test_constant_potts_resolution_controls_how_fine_the_partition_is(): void
    {
        $graph = GraphFixture::load('zachary')->graph();

        $coarse = Leiden::constantPotts(resolution: 0.02, seed: 42)->partition($graph);
        $fine = Leiden::constantPotts(resolution: 0.5, seed: 42)->partition($graph);

        self::assertLessThan(
            $fine->count(),
            $coarse->count(),
            'a lower CPM resolution must not produce more communities than a higher one',
        );
    }
}
