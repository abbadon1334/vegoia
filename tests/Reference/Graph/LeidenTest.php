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

    /**
     * The guarantee holds for every objective, not just the default.
     *
     * This caught a real bug. With RBER the search could stall -- refinement
     * grouping nothing, so aggregation returned the same graph and the loop
     * spun to its iteration cap -- and exiting that way skips the convergence
     * condition the guarantee normally rests on. It emerged only once a third
     * objective existed to exercise it.
     *
     * @return iterable<string, array{string, callable(int): Leiden}>
     */
    public static function objectives(): iterable
    {
        yield 'modularity' => ['modularity', static fn (int $seed): Leiden => Leiden::modularity(seed: $seed)];
        yield 'cpm' => ['cpm', static fn (int $seed): Leiden => Leiden::constantPotts(0.1, $seed)];
        yield 'rber' => ['rber', static fn (int $seed): Leiden => Leiden::erdosRenyiPotts(1.0, $seed)];
    }

    /** @return iterable<string, array{string, string, callable(int): Leiden}> */
    public static function fixturesAndObjectives(): iterable
    {
        foreach (GraphFixture::names() as $name) {
            foreach (self::objectives() as [$objective, $factory]) {
                yield "{$name}/{$objective}" => [$name, $objective, $factory];
            }
        }
    }

    /** @param callable(int): Leiden $factory */
    #[DataProvider('fixturesAndObjectives')]
    public function test_communities_are_connected_whatever_the_objective(
        string $name,
        string $objective,
        callable $factory,
    ): void {
        $graph = GraphFixture::load($name)->graph();

        for ($seed = 1; $seed <= 10; $seed++) {
            foreach ($factory($seed)->partition($graph)->communities() as $index => $members) {
                self::assertTrue(
                    Connectivity::inducesConnectedSubgraph($graph, $members),
                    "{$name}/{$objective}: seed {$seed} left community {$index} internally disconnected",
                );
            }
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
    public function test_it_reaches_the_best_modularity_leidenalg_reaches(string $name): void
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

        // Best against best. This asked for the bottom of the reference band
        // until leidenalg 0.12 made the point that the bottom is the wrong
        // bar: its worst seed on the 4x4 grid fell from 0.4167 to 0.3333
        // between releases, which would have silently relaxed this assertion
        // by a fifth. The top of the band did not move on any fixture -- an
        // optimum is a property of the graph, while a worst case is a property
        // of one library's random order -- so comparing best to best is both
        // the stricter test and the stable one.
        //
        // Measured, not hoped for: this implementation reaches leidenalg's
        // best on all eleven fixtures, and does it in ten seeds against the
        // reference's fifty.
        self::assertGreaterThanOrEqual(
            $envelope['max'] - 1.0e-9,
            $best,
            sprintf(
                "%s: best modularity over 10 seeds was %.9f, short of leidenalg's best of %.9f "
                . '(reference band [%.9f, %.9f])',
                $name,
                $best,
                $envelope['max'],
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

    /**
     * Refinement must actually refine.
     *
     * This test exists because its absence was a real hole. Every other
     * assertion in this file passed with the refinement phase deleted -- with
     * Leiden degraded to plain Louvain -- because on graphs whose structure is
     * clear enough, Louvain reaches the same partition by a different route.
     * The connectivity guarantee is real but does not discriminate here: none
     * of the fixtures is one Louvain actually fails on.
     *
     * So the difference is asserted where it is unambiguous: refinement splits
     * a community into well-connected parts and aggregates *those*, so at the
     * first level the refined partition must be strictly finer than the
     * communities local moving found. Skip refinement and the two are equal at
     * every level.
     */
    #[DataProvider('fixturesWithStructure')]
    public function test_refinement_splits_communities_before_aggregating(string $name): void
    {
        $graph = GraphFixture::load($name)->graph();

        [, $trace] = Leiden::modularity(seed: 42)->partitionWithTrace($graph);

        self::assertNotSame([], $trace, "{$name}: the algorithm never aggregated");

        $refinedFiner = false;

        foreach ($trace as $level) {
            self::assertGreaterThanOrEqual(
                $level['communities'],
                $level['refined'],
                "{$name}: level {$level['level']} refined into fewer parts than there are "
                . 'communities, which is impossible for a sub-partition',
            );

            self::assertLessThanOrEqual(
                $level['nodes'],
                $level['refined'],
                "{$name}: level {$level['level']} refined into more parts than there are nodes",
            );

            if ($level['refined'] > $level['communities']) {
                $refinedFiner = true;
            }
        }

        self::assertTrue(
            $refinedFiner,
            "{$name}: refinement never split a single community at any level -- "
            . 'this is Louvain, not Leiden',
        );
    }

    /**
     * The fixtures on which refinement demonstrably has something to split,
     * established by running it rather than assumed.
     *
     * Most fixtures are not on this list, and for good reason: where local
     * moving already lands on communities that cannot be divided into
     * well-connected parts -- ring_of_cliques, whose cliques are each already
     * one -- refinement correctly finds nothing, and demanding a split there
     * would assert something false.
     *
     * @return iterable<string, array{string}>
     */
    public static function fixturesWithStructure(): iterable
    {
        foreach (['zachary', 'lesmis', 'davis', 'grid_4x4'] as $name) {
            yield $name => [$name];
        }
    }

    /**
     * The gamma-connectivity condition, checked on the parts refinement
     * produced.
     *
     * Refinement may only merge into a part that is already well connected to
     * the rest of its community:
     *
     *     E(C, S \ C)  >=  y * ||C|| * (||S|| - ||C||) / 2m
     *
     * where ||.|| is strength under modularity. This is what stops a
     * community being welded together through a node that does not belong in
     * it, and it is the condition the whole guarantee rests on.
     *
     * Asserted here because deleting the check left every other test green:
     * dropping it does not produce disconnected communities, it quietly
     * weakens a guarantee, which is exactly the kind of regression that
     * survives a suite that only looks at outputs.
     */
    #[DataProvider('fixturesWithStructure')]
    public function test_refined_parts_satisfy_gamma_connectivity(string $name): void
    {
        $resolution = 1.0;
        [, $trace] = Leiden::modularity($resolution, seed: 42)
            ->partitionWithTrace(GraphFixture::load($name)->graph());

        $checked = 0;

        foreach ($trace as $level) {
            $graph = $level['graph'];
            $twoM = $graph->totalEndpointWeight();

            foreach ($level['partition']->communities() as $community) {
                $inside = [];
                $subsetStrength = 0.0;

                foreach ($community as $node) {
                    $inside[$node] = true;
                    $subsetStrength += $graph->strength($node);
                }

                // Group this community's nodes by the refined part they landed in.
                $parts = [];

                foreach ($community as $node) {
                    $parts[$level['parts']->communityOf($node)][] = $node;
                }

                foreach ($parts as $members) {
                    // A part that is still the whole community has no rest to
                    // be connected to, and the condition is vacuous.
                    if (count($members) === count($community)) {
                        continue;
                    }

                    $partStrength = 0.0;
                    $toRest = 0.0;

                    foreach ($members as $node) {
                        $partStrength += $graph->strength($node);

                        foreach ($graph->neighbours($node) as $neighbour => $weight) {
                            if (isset($inside[$neighbour]) && ! in_array($neighbour, $members, true)) {
                                $toRest += $weight;
                            }
                        }
                    }

                    $threshold = $resolution * $partStrength * ($subsetStrength - $partStrength) / $twoM;

                    self::assertGreaterThanOrEqual(
                        $threshold - 1.0e-9,
                        $toRest,
                        sprintf(
                            '%s level %d: a refined part of %d nodes has weight %.6f to the rest '
                            . 'of its community but gamma-connectivity requires %.6f',
                            $name,
                            $level['level'],
                            count($members),
                            $toRest,
                            $threshold,
                        ),
                    );

                    $checked++;
                }
            }
        }

        self::assertGreaterThan(0, $checked, "{$name}: no refined part was actually checked");
    }

    public function test_the_trace_describes_the_aggregation_it_performed(): void
    {
        $graph = GraphFixture::load('lesmis')->graph();

        [$partition, $trace] = Leiden::modularity(seed: 42)->partitionWithTrace($graph);

        self::assertNotSame([], $trace);
        self::assertSame(0, $trace[0]['level']);
        self::assertSame($graph->order(), $trace[0]['nodes'], 'level 0 sees the original graph');

        // Each level runs on the parts the previous one refined into.
        for ($i = 1; $i < count($trace); $i++) {
            self::assertSame(
                $trace[$i - 1]['refined'],
                $trace[$i]['nodes'],
                "level {$i} must run on the previous level's refined parts",
            );
        }

        self::assertLessThanOrEqual($trace[0]['nodes'], $partition->count());
    }

    public function test_an_empty_graph_traces_nothing(): void
    {
        [$partition, $trace] = Leiden::modularity()->partitionWithTrace(Graph::undirected(0));

        self::assertSame(0, $partition->count());
        self::assertSame([], $trace);
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

    /**
     * RBER optimises, not merely scores. It reaches the objective through the
     * binding step rather than the constructor, so a run that silently used an
     * unbound density -- or threw -- would be a live failure mode.
     */
    public function test_erdos_renyi_potts_can_be_optimised(): void
    {
        $graph = GraphFixture::load('zachary')->graph();

        $partition = Leiden::erdosRenyiPotts(resolution: 1.0, seed: 42)->partition($graph);

        self::assertGreaterThan(1, $partition->count());
        self::assertSame($graph->order(), $partition->order());

        foreach ($partition->communities() as $members) {
            self::assertTrue(Connectivity::inducesConnectedSubgraph($graph, $members));
        }
    }

    /** leidenalg's RBConfiguration is modularity with a resolution parameter. */
    public function test_reichardt_bornholdt_is_modularity_under_another_name(): void
    {
        $graph = GraphFixture::load('zachary')->graph();

        self::assertTrue(
            Leiden::reichardtBornholdt(0.7, seed: 3)->partition($graph)
                ->equals(Leiden::modularity(0.7, seed: 3)->partition($graph)),
        );
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
