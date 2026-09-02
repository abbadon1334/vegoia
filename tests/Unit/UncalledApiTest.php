<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\TestCase;
use Vegoia\Graph\Community\Quality\Modularity;
use Vegoia\Graph\Graph;
use Vegoia\Graph\NodeIndex;
use Vegoia\Graph\Partition;
use Vegoia\Stats\Descriptive;
use Vegoia\Stats\Distribution\ChiSquared;
use Vegoia\Stats\Distribution\FisherSnedecor;
use Vegoia\Stats\Distribution\Normal;
use Vegoia\Stats\Distribution\StudentT;

/**
 * Published methods the suite had never once executed.
 *
 * Found by reading the coverage report for methods with a hit count of zero
 * rather than for lines, which is a different and more uncomfortable question:
 * not "is this branch taken" but "has anyone ever called this at all". These
 * eight were the answer, and a method nobody has run is a method whose
 * behaviour is a guess -- it type-checks, and that is all anyone knows.
 *
 * Each is asserted against what its name promises and against the neighbouring
 * method it is easily confused with, since the failure that survives review is
 * an accessor returning the plausible wrong thing.
 */
#[CoversClass(Graph::class)]
#[CoversClass(NodeIndex::class)]
#[CoversClass(Partition::class)]
#[CoversClass(Descriptive::class)]
final class UncalledApiTest extends TestCase
{
    /**
     * A graph is empty when it has no nodes, not when it has no edges.
     *
     * The distinction is the whole content of the method: an edgeless graph
     * on 5 nodes is a perfectly good graph with five communities in it, and
     * treating it as empty would make every caller that guards with isEmpty()
     * silently skip it.
     */
    public function test_a_graph_is_empty_when_it_has_no_nodes(): void
    {
        self::assertTrue(Graph::undirected(0)->isEmpty());
        self::assertTrue(Graph::directed(0)->isEmpty());

        self::assertFalse(Graph::undirected(5)->isEmpty(), 'no edges is not the same as no nodes');
        self::assertFalse(Graph::undirected(1)->isEmpty());
        self::assertFalse(Graph::undirected(2, [[0, 1]])->isEmpty());
    }

    /**
     * The identifiers, in the order the nodes were assigned.
     *
     * That order is the contract: identifiers()[$n] must name the same node
     * identifierFor($n) does, or every label a caller reads back is off by
     * however many times the index happened to reorder itself.
     */
    public function test_the_index_lists_identifiers_in_node_order(): void
    {
        $index = new NodeIndex();

        self::assertSame([], $index->identifiers(), 'an empty index lists nothing');

        foreach (['zeta', 'alpha', 'mu'] as $identifier) {
            $index->add($identifier);
        }

        // Insertion order, not sorted: 'alpha' is node 1 because it arrived
        // second, and sorting here would silently renumber every node.
        self::assertSame(['zeta', 'alpha', 'mu'], $index->identifiers());

        // Re-adding an existing identifier must not append a second entry.
        $index->add('alpha');

        self::assertSame(['zeta', 'alpha', 'mu'], $index->identifiers());
        self::assertCount($index->count(), $index->identifiers());

        foreach (['zeta', 'alpha', 'mu'] as $node => $identifier) {
            self::assertSame($identifier, $index->identifierFor($node));
            self::assertSame($node, $index->nodeFor($identifier));
        }
    }

    /**
     * One community holding everything -- the partition Leiden starts from
     * when asked whether splitting is worth anything at all, and the one whose
     * modularity is zero by construction.
     */
    public function test_a_single_community_holds_every_node(): void
    {
        $partition = Partition::single(6);

        self::assertSame([0, 0, 0, 0, 0, 0], $partition->membership());
        self::assertSame(1, $partition->count());
        self::assertSame(6, $partition->order());
        self::assertSame([6], $partition->sizes());
        self::assertFalse($partition->hasUnassigned());

        // Not the same thing as singletons(), which is the other degenerate
        // partition and the easy one to return by mistake.
        self::assertNotSame(Partition::singletons(6)->membership(), $partition->membership());

        // Modularity of the whole graph as one community is exactly zero: the
        // edge fraction is 1 and the expected fraction is 1, and they cancel.
        $graph = Graph::undirected(4, [[0, 1], [1, 2], [2, 3], [3, 0]]);

        self::assertEqualsWithDelta(0.0, new Modularity()->of($graph, Partition::single(4)), 1.0e-15);
    }

    /**
     * On no nodes there is no community to put them in, so the count is zero
     * rather than one empty community.
     */
    public function test_a_single_community_of_nothing_is_no_community(): void
    {
        $partition = Partition::single(0);

        self::assertSame([], $partition->membership());
        self::assertSame(0, $partition->count());
        self::assertSame(0, $partition->order());
    }

    /** Emptiness is about having no values, which zero is not. */
    public function test_a_summary_is_empty_only_with_no_values(): void
    {
        self::assertTrue(Descriptive::of([])->isEmpty());

        self::assertFalse(Descriptive::of([0.0])->isEmpty(), 'a single zero is a value');
        self::assertFalse(Descriptive::of([0.0, 0.0])->isEmpty());
        self::assertFalse(Descriptive::of([-1.5, 2.0])->isEmpty());

        self::assertSame(0, Descriptive::of([])->count());
        self::assertSame(1, Descriptive::of([0.0])->count());
    }

    /**
     * The ends of the support, which are where a quantile stops being a
     * number.
     *
     * These are the answers the generic inverse reaches through infimum() and
     * its opposite, and they are pinned here because Normal and Student's t
     * both override the inverse with a closed form and never take that path --
     * so the values are reachable and the code producing them is not the code
     * a reader would expect. The distinction that matters is the lower end:
     * unbounded for the symmetric pair, exactly zero for the two that live on
     * the non-negative half.
     */
    public function test_the_quantiles_at_the_ends_of_each_support(): void
    {
        foreach (['normal' => new Normal(), 't' => new StudentT(5.0)] as $name => $symmetric) {
            self::assertSame(-INF, $symmetric->quantile(0.0), $name);
            self::assertSame(INF, $symmetric->quantile(1.0), $name);
            self::assertSame(INF, $symmetric->upperQuantile(0.0), $name);
            self::assertSame(-INF, $symmetric->upperQuantile(1.0), $name);
        }

        foreach (['chi2' => new ChiSquared(3.0), 'F' => new FisherSnedecor(4.0, 7.0)] as $name => $positive) {
            self::assertSame(0.0, $positive->quantile(0.0), "{$name}: the support starts at zero");
            self::assertSame(INF, $positive->quantile(1.0), $name);
            self::assertSame(INF, $positive->upperQuantile(0.0), $name);
            self::assertSame(0.0, $positive->upperQuantile(1.0), "{$name}: and ends there from above");
        }

        // A shifted normal moves its ends nowhere: infinity is not relative
        // to the mean.
        self::assertSame(-INF, new Normal(100.0, 15.0)->quantile(0.0));
        self::assertSame(INF, new Normal(100.0, 15.0)->quantile(1.0));
    }
}
