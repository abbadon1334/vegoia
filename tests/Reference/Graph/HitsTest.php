<?php

declare(strict_types=1);

namespace Vegoia\Tests\Reference\Graph;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Vegoia\Graph\Centrality\Hits;
use Vegoia\Graph\Graph;
use Vegoia\Tests\Support\GraphFixture;

/**
 * HITS on directed graphs.
 *
 * Two findings shaped this file, and both are about what NOT to assert.
 *
 * HITS says nothing on an undirected graph: A equals its transpose, so hubs
 * and authorities are the same vector and the measure collapses to something
 * eigenvector centrality already provides.
 *
 * More importantly, the scores are the principal eigenvectors of A A^T and
 * A^T A, so they exist only when the leading eigenvalue is simple. On a
 * directed cycle A A^T is the identity -- every eigenvalue is 1, every vector
 * is an eigenvector. On a bow tie the largest has multiplicity three. There
 * every implementation returns something, and what it returns is an accident
 * of the solver: networkx at tol=1e-14 returns negative scores and scores
 * above 1, neither of which HITS can produce.
 *
 * So the fixtures record values only where the spectral gap says the answer is
 * unique, and the degenerate shapes are tested by property instead.
 */
#[CoversClass(Hits::class)]
#[Group('reference')]
final class HitsTest extends TestCase
{
    /** @return iterable<string, array{string}> */
    public static function directedFixtures(): iterable
    {
        foreach (GraphFixture::directedNames() as $name) {
            yield $name => [$name];
        }
    }

    #[DataProvider('directedFixtures')]
    public function test_hub_and_authority_scores_match_the_principal_eigenvectors(string $name): void
    {
        $fixture = GraphFixture::directed($name);
        $expected = $fixture->expectedHits();

        if ($expected === null) {
            self::markTestSkipped("{$name}: the leading eigenvalue is not simple, so HITS is undefined");
        }

        [$hubs, $authorities] = $expected;
        $scores = (new Hits())->of($fixture->directedGraph());

        foreach ($hubs as $node => $value) {
            self::assertEqualsWithDelta($value, $scores['hubs'][$node], 1.0e-9, "{$name}: hub {$node}");
        }

        foreach ($authorities as $node => $value) {
            self::assertEqualsWithDelta($value, $scores['authorities'][$node], 1.0e-9, "{$name}: authority {$node}");
        }
    }

    #[DataProvider('directedFixtures')]
    public function test_the_scores_are_non_negative_and_sum_to_one(string $name): void
    {
        // Holds on every graph, degenerate or not: whatever eigenvector the
        // iteration settles on, it cannot be negative and it is normalised.
        $scores = (new Hits())->of(GraphFixture::directed($name)->directedGraph());

        foreach (['hubs', 'authorities'] as $kind) {
            foreach ($scores[$kind] as $node => $value) {
                self::assertGreaterThanOrEqual(0.0, $value, "{$name}: negative {$kind} at {$node}");
            }

            self::assertEqualsWithDelta(1.0, array_sum($scores[$kind]), 1.0e-9, "{$name}: {$kind} sum");
        }
    }

    /**
     * Kleinberg's own shape. Nodes 0-2 only point, nodes 3-5 only receive, so
     * the separation is total and needs no reference to verify.
     */
    public function test_pure_hubs_and_pure_authorities_separate_completely(): void
    {
        $graph = Graph::directed(6, [
            [0, 3, 1.0], [0, 4, 1.0], [0, 5, 1.0],
            [1, 3, 1.0], [1, 4, 1.0],
            [2, 4, 1.0], [2, 5, 1.0],
        ]);

        $scores = (new Hits())->of($graph);

        for ($node = 0; $node <= 2; $node++) {
            self::assertGreaterThan(0.0, $scores['hubs'][$node], "node {$node} points at things");
            self::assertEqualsWithDelta(0.0, $scores['authorities'][$node], 1.0e-12, "nothing points at {$node}");
        }

        for ($node = 3; $node <= 5; $node++) {
            self::assertEqualsWithDelta(0.0, $scores['hubs'][$node], 1.0e-12, "node {$node} points at nothing");
            self::assertGreaterThan(0.0, $scores['authorities'][$node], "things point at {$node}");
        }
    }

    /**
     * On an undirected graph the adjacency matrix is symmetric, so A A^T and
     * A^T A are the same matrix and the two scores must coincide. Worth
     * asserting because it is the reason HITS adds nothing there.
     */
    public function test_the_two_scores_coincide_on_an_undirected_graph(): void
    {
        $scores = (new Hits())->of(GraphFixture::load('zachary')->graph());

        foreach ($scores['hubs'] as $node => $hub) {
            self::assertEqualsWithDelta($hub, $scores['authorities'][$node], 1.0e-9, "node {$node}");
        }
    }

    public function test_degenerate_graphs_still_return_a_valid_distribution(): void
    {
        // A directed cycle: A A^T is the identity, so no eigenvector is
        // preferred. The result must still be a usable distribution rather
        // than NaN or a negative vector.
        $cycle = Graph::directed(4, [[0, 1, 1.0], [1, 2, 1.0], [2, 3, 1.0], [3, 0, 1.0]]);
        $scores = (new Hits())->of($cycle);

        self::assertEqualsWithDelta(1.0, array_sum($scores['hubs']), 1.0e-9);
        self::assertEqualsWithDelta(1.0, array_sum($scores['authorities']), 1.0e-9);

        foreach ($scores['hubs'] as $value) {
            self::assertTrue(is_finite($value));
            self::assertGreaterThanOrEqual(0.0, $value);
        }
    }

    public function test_an_edgeless_graph_yields_a_uniform_distribution(): void
    {
        $scores = (new Hits())->of(Graph::directed(4));

        self::assertSame([0.25, 0.25, 0.25, 0.25], $scores['hubs']);
        self::assertSame([0.25, 0.25, 0.25, 0.25], $scores['authorities']);
    }

    public function test_an_empty_graph_yields_empty_vectors(): void
    {
        $scores = (new Hits())->of(Graph::directed(0));

        self::assertSame([], $scores['hubs']);
        self::assertSame([], $scores['authorities']);
    }
}
