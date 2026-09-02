<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit\Graph\Community\Quality;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;
use Vegoia\Graph\Community\Leiden;
use Vegoia\Graph\Community\Quality\ConstantPotts;
use Vegoia\Graph\Community\Quality\ErdosRenyiPotts;
use Vegoia\Graph\Community\Quality\Modularity;
use Vegoia\Graph\Community\Quality\QualityFunction;
use Vegoia\Graph\Graph;
use Vegoia\Graph\Partition;

/**
 * The half of each objective that Leiden's fast path replaces.
 *
 * Leiden inlines the gain arithmetic for the three built-in objectives, and
 * dispatches through the interface for everything else -- selected on the
 * exact class, so that a caller's own objective always takes the general
 * path. That leaves gain(), measure() and connectivityThreshold() on
 * Modularity, ConstantPotts and ErdosRenyiPotts as code the whole test suite
 * ran past without once executing: two implementations of one formula, one of
 * them unexercised.
 *
 * Two things are checked here, and neither is sufficient alone. First, that
 * each gain() really is proportional to the change it claims to predict --
 * measured against of(), which the reference suite pins to igraph. Second,
 * that the fast path and the interface path reach the same partition. The
 * second alone would pass if both were wrong in the same way; the first alone
 * would not notice the inlined copy drifting.
 */
#[CoversClass(Modularity::class)]
#[CoversClass(ConstantPotts::class)]
#[CoversClass(ErdosRenyiPotts::class)]
final class ObjectiveContractTest extends TestCase
{
    /**
     * Two triangles joined by a single edge, plus a pendant. No self-loops and
     * no parallel edges, so the local quantities this test computes for itself
     * out of the Graph API need no special cases to be right.
     */
    private static function graph(): Graph
    {
        return Graph::undirected(7, [
            [0, 1, 1.0], [1, 2, 2.0], [0, 2, 1.5],
            [3, 4, 1.0], [4, 5, 3.0], [3, 5, 1.0],
            [2, 3, 0.5],
            [5, 6, 2.0],
        ]);
    }

    /** @return iterable<string, array{QualityFunction}> */
    public static function objectives(): iterable
    {
        yield 'modularity' => [new Modularity()];
        yield 'modularity at resolution 0.4' => [new Modularity(0.4)];
        yield 'modularity at resolution 2.5' => [new Modularity(2.5)];
        yield 'CPM' => [new ConstantPotts(0.25)];
        yield 'CPM at resolution 1.0' => [new ConstantPotts()];
        yield 'RBER' => [new ErdosRenyiPotts(0.8)];
    }

    /**
     * gain() must be proportional to the change in of(), with one positive
     * constant of proportionality for the whole graph.
     *
     * That is the entire contract: the interface promises a value
     * "proportional to the true delta, not equal to it", and every comparison
     * Leiden makes is valid only if the constant really is constant. A gain
     * that got the resolution term wrong, or dropped a factor that varies with
     * the candidate community, still produces plausible-looking numbers -- it
     * just stops ranking the candidates the way the objective does. Here the
     * ratio is computed for each of 30 moves and every one must agree with the
     * first to twelve digits.
     */
    #[DataProvider('objectives')]
    public function test_the_gain_is_proportional_to_the_change_it_predicts(QualityFunction $objective): void
    {
        $graph = self::graph();
        $objective = $objective->boundTo($graph);

        // Three communities, so every node has two elsewhere to move to.
        /** @var list<int> $membership */
        $membership = [0, 0, 0, 1, 1, 1, 2];

        $ratios = [];

        foreach ($graph->nodes() as $node) {
            $from = $membership[$node];

            foreach ([0, 1, 2] as $to) {
                if ($to === $from) {
                    continue;
                }

                $before = $objective->of($graph, Partition::fromMembership($membership));

                $moved = [];

                foreach ($membership as $other => $community) {
                    $moved[] = $other === $node ? $to : $community;
                }

                $after = $objective->of($graph, Partition::fromMembership($moved));

                $predicted = $objective->gain(...self::locals($graph, $node, $membership, $to))
                    - $objective->gain(...self::locals($graph, $node, $membership, $from));

                $actual = $after - $before;

                // A move that changes nothing carries no information about the
                // constant; it must, however, be predicted as nothing.
                if (abs($actual) < 1.0e-12 && abs($predicted) < 1.0e-12) {
                    continue;
                }

                self::assertNotEqualsWithDelta(
                    0.0,
                    $predicted,
                    1.0e-12,
                    "moving node {$node} to community {$to} changes the score by {$actual}, "
                    . 'and the gain predicted no change at all',
                );

                $ratios[] = ['ratio' => $actual / $predicted, 'node' => $node, 'to' => $to];
            }
        }

        self::assertGreaterThanOrEqual(6, count($ratios), 'too few moves to establish a constant');

        $constant = $ratios[0]['ratio'];

        self::assertGreaterThan(
            0.0,
            $constant,
            'the constant is negative, so every comparison Leiden makes is backwards',
        );

        foreach ($ratios as $seen) {
            self::assertEqualsWithDelta(
                $constant,
                $seen['ratio'],
                1.0e-12 * $constant,
                sprintf(
                    'moving node %d to community %d scales by %.15g, the first move scaled by %.15g -- '
                    . 'the factor depends on the candidate, so gain() does not rank them as of() does',
                    $seen['node'],
                    $seen['to'],
                    $seen['ratio'],
                    $constant,
                ),
            );
        }
    }

    /**
     * The local quantities gain() is given, computed out of the Graph API
     * rather than out of Leiden's bookkeeping -- the point being to check that
     * bookkeeping, not to reuse it.
     *
     * The node is excluded from the community totals in both directions: from
     * the one it is leaving because the comparison is against the state
     * without it, and from the one it is joining because it is not there yet.
     *
     * @param list<int> $membership
     *
     * @return array{float, float, float, float, float, float}
     */
    private static function locals(Graph $graph, int $node, array $membership, int $community): array
    {
        $weightToCommunity = 0.0;

        foreach ($graph->neighbours($node) as $neighbour => $weight) {
            if ($neighbour !== $node && $membership[$neighbour] === $community) {
                $weightToCommunity += $weight;
            }
        }

        $communityStrength = 0.0;
        $communitySize = 0.0;

        foreach ($graph->nodes() as $other) {
            if ($other !== $node && $membership[$other] === $community) {
                $communityStrength += $graph->strength($other);
                $communitySize++;
            }
        }

        return [
            $weightToCommunity,
            $graph->strength($node),
            1.0,
            $communityStrength,
            $communitySize,
            $graph->totalEndpointWeight(),
        ];
    }

    /**
     * What each objective measures a set by. Modularity counts edge ends, the
     * Potts family counts nodes -- which is the whole difference between them,
     * and the reason CPM has no resolution limit.
     */
    public function test_each_objective_measures_a_set_by_its_own_quantity(): void
    {
        self::assertSame(7.5, new Modularity()->measure(7.5, 3.0));
        self::assertSame(7.5, new Modularity(2.0)->measure(7.5, 3.0), 'resolution does not enter the measure');

        self::assertSame(3.0, new ConstantPotts()->measure(7.5, 3.0));
        self::assertSame(3.0, new ErdosRenyiPotts()->measure(7.5, 3.0));
    }

    /**
     * The gamma-connectivity threshold of the Leiden paper: a part is well
     * connected to its subset when the weight between them reaches
     * gamma * |part| * (|subset| - |part|), normalised by 2m for modularity
     * because that is the scale its own gain is on.
     *
     * The numbers are worked by hand rather than read off the implementation.
     */
    public function test_the_connectivity_threshold_follows_the_paper(): void
    {
        // gamma 1, part 4, subset 10, 2m = 20:  1 * 4 * 6 / 20
        self::assertEqualsWithDelta(1.2, new Modularity()->connectivityThreshold(4.0, 10.0, 20.0), 1.0e-15);
        // gamma 0.5 halves it.
        self::assertEqualsWithDelta(0.6, new Modularity(0.5)->connectivityThreshold(4.0, 10.0, 20.0), 1.0e-15);

        // CPM does not divide: 0.25 * 3 * 5.
        self::assertEqualsWithDelta(3.75, new ConstantPotts(0.25)->connectivityThreshold(3.0, 8.0, 20.0), 1.0e-15);

        // RBER scales CPM's by the density it bound. On a 5-node graph with
        // total weight 4, the density is 4 / (5 * 4 / 2) = 0.4, so the
        // threshold is 0.4 of CPM's at the same resolution.
        $graph = Graph::undirected(5, [[0, 1], [1, 2], [2, 3], [3, 4]]);
        $rber = new ErdosRenyiPotts(0.25)->boundTo($graph);

        self::assertEqualsWithDelta(0.4, $rber->density(), 1.0e-15);
        self::assertEqualsWithDelta(1.5, $rber->connectivityThreshold(3.0, 8.0, 20.0), 1.0e-15);

        // An unbound RBER has no density to scale by and behaves as CPM does,
        // rather than silently returning zero.
        self::assertEqualsWithDelta(3.75, new ErdosRenyiPotts(0.25)->connectivityThreshold(3.0, 8.0, 20.0), 1.0e-15);
    }

    /**
     * A graph with no edges has no scale to divide by, and modularity says so
     * instead of dividing by zero.
     */
    public function test_modularity_on_a_graph_with_no_edge_ends(): void
    {
        self::assertSame(0.0, new Modularity()->connectivityThreshold(4.0, 10.0, 0.0));
        self::assertSame(0.0, new Modularity()->gain(0.0, 0.0, 1.0, 0.0, 0.0, 0.0));
    }

    /**
     * The inlined path and the interface path must agree exactly.
     *
     * Leiden selects the inlined arithmetic by exact class, so wrapping an
     * objective in a delegate that changes nothing but the class name sends
     * the identical formula down the other branch. Same seed, same graph, same
     * objective: the partitions must be identical node for node. Anything less
     * means the two copies of the formula have drifted, and which one a caller
     * gets depends on whether they happened to pass a built-in class.
     *
     * @return iterable<string, array{QualityFunction}>
     */
    public static function fastPathObjectives(): iterable
    {
        yield 'modularity' => [new Modularity()];
        yield 'modularity at resolution 0.5' => [new Modularity(0.5)];
        yield 'modularity at resolution 3.0' => [new Modularity(3.0)];
        yield 'CPM' => [new ConstantPotts(0.05)];
        yield 'CPM at a resolution that splits' => [new ConstantPotts(0.6)];
        yield 'RBER' => [new ErdosRenyiPotts(0.5)];
        yield 'RBER at a high resolution' => [new ErdosRenyiPotts(3.0)];
    }

    #[DataProvider('fastPathObjectives')]
    public function test_the_inlined_gain_agrees_with_the_interface(QualityFunction $objective): void
    {
        $graph = self::graph();

        for ($seed = 1; $seed <= 8; $seed++) {
            $inlined = new Leiden($objective, $seed)->partition($graph);
            $general = new Leiden(new DelegatingObjective($objective), $seed)->partition($graph);

            self::assertSame(
                $inlined->membership(),
                $general->membership(),
                sprintf(
                    'seed %d: the inlined arithmetic and %s::gain() reached different partitions',
                    $seed,
                    $objective::class,
                ),
            );
        }
    }

    /**
     * The same, on a graph large enough for the refinement phase to have real
     * work to do -- that is where measure() and connectivityThreshold() are
     * called, and the small graph above barely exercises them.
     */
    public function test_the_two_paths_agree_on_a_graph_with_real_refinement(): void
    {
        // A ring of six triangles: refinement has to decide, repeatedly,
        // whether a part is well enough connected to the rest of its subset.
        $edges = [];

        for ($t = 0; $t < 6; $t++) {
            $a = $t * 3;
            $edges[] = [$a, $a + 1, 1.0];
            $edges[] = [$a + 1, $a + 2, 1.0];
            $edges[] = [$a, $a + 2, 1.0];
            $edges[] = [$a + 2, ($a + 3) % 18, 0.4];
        }

        $graph = Graph::undirected(18, $edges);

        foreach ([new Modularity(), new ConstantPotts(0.3), new ErdosRenyiPotts(1.5)] as $objective) {
            for ($seed = 1; $seed <= 5; $seed++) {
                self::assertSame(
                    new Leiden($objective, $seed)->partition($graph)->membership(),
                    new Leiden(new DelegatingObjective($objective), $seed)->partition($graph)->membership(),
                    sprintf('%s, seed %d', $objective::class, $seed),
                );
            }
        }
    }
}

/**
 * Forwards every call to a real objective, changing nothing but the class.
 *
 * That is enough to leave Leiden's fast path -- which matches on the exact
 * class precisely so that somebody else's objective is never mistaken for a
 * built-in one -- and take the interface path instead.
 */
final readonly class DelegatingObjective implements QualityFunction
{
    public function __construct(private QualityFunction $inner)
    {
    }

    public function of(Graph $graph, Partition $partition): float
    {
        return $this->inner->of($graph, $partition);
    }

    public function resolution(): float
    {
        return $this->inner->resolution();
    }

    public function boundTo(Graph $graph): self
    {
        return new self($this->inner->boundTo($graph));
    }

    public function gain(
        float $weightToCommunity,
        float $nodeStrength,
        float $nodeSize,
        float $communityStrength,
        float $communitySize,
        float $totalEndpointWeight,
    ): float {
        return $this->inner->gain(
            $weightToCommunity,
            $nodeStrength,
            $nodeSize,
            $communityStrength,
            $communitySize,
            $totalEndpointWeight,
        );
    }

    public function measure(float $strength, float $size): float
    {
        return $this->inner->measure($strength, $size);
    }

    public function connectivityThreshold(
        float $partMeasure,
        float $subsetMeasure,
        float $totalEndpointWeight,
    ): float {
        return $this->inner->connectivityThreshold($partMeasure, $subsetMeasure, $totalEndpointWeight);
    }
}
