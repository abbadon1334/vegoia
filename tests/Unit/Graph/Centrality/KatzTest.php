<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit\Graph\Centrality;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\TestCase;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Graph\Centrality\Katz;
use Vegoia\Graph\Graph;

/**
 * The safety rail in front of Katz, and where it used to stop working.
 *
 * `isConvergent()` is the predicate a caller asks *before* trusting Katz
 * scores, and it had no test at all. Testing it turned up a real limit rather
 * than a missing assertion.
 */
#[CoversClass(Katz::class)]
final class KatzTest extends TestCase
{
    private static function path(int $nodes): Graph
    {
        $edges = [];

        for ($i = 0; $i < $nodes - 1; $i++) {
            $edges[] = [$i, $i + 1];
        }

        return Graph::undirected($nodes, $edges);
    }

    /**
     * A long path is where the estimate cannot settle, and the fallback is
     * what keeps Katz usable there.
     *
     * The eigenvalues of a path crowd together as it lengthens: on 201 nodes
     * the largest two are 1.999758 and 1.999033, and after the shift the
     * power iteration's ratio is 0.99982, which needs about 152 000 steps to
     * reach twelve digits. It used to refuse outright at a thousand. Raising
     * the ceiling is not the answer either -- a thousand-node path takes 34
     * seconds to converge.
     *
     * So an estimate that has not settled falls back on the maximum strength,
     * which bounds the spectral radius from above and therefore only ever
     * rejects alphas that would have been accepted. On these graphs that is
     * 0.012% of the range, because a path's maximum degree is 2 and its
     * spectral radius is 1.9998.
     */
    public function test_a_long_path_still_gets_an_answer(): void
    {
        $radius = Katz::spectralRadius(self::path(201));

        // The safe bound, not the true 1.999758 -- and never below it, which
        // is the property that makes the fallback safe rather than merely
        // convenient.
        self::assertGreaterThanOrEqual(1.999758, $radius);
        self::assertLessThanOrEqual(2.0, $radius);

        $scores = new Katz(alpha: 0.4)->of(self::path(201));

        self::assertCount(201, $scores);
        self::assertGreaterThan($scores[0], $scores[100], 'the middle of a path is better connected');
    }

    /** Where the iteration does settle, the estimate is the real thing. */
    public function test_an_ordinary_graph_gets_the_true_radius(): void
    {
        // A star: the spectral radius is exactly sqrt(n - 1), and the maximum
        // strength is n - 1, so an implementation that had quietly fallen back
        // on the bound would be caught here by a factor of three.
        $star = Graph::undirected(10, [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9]]);

        self::assertEqualsWithDelta(3.0, Katz::spectralRadius($star), 1.0e-9);
        self::assertEqualsWithDelta(1.0 / 3.0, Katz::criticalAlpha($star), 1.0e-9);
    }

    /**
     * The predicate a caller is supposed to ask first, and which nothing was
     * asking.
     */
    public function test_it_says_in_advance_whether_alpha_will_converge(): void
    {
        $star = Graph::undirected(5, [[0, 1], [0, 2], [0, 3], [0, 4]]);
        $critical = Katz::criticalAlpha($star);

        self::assertTrue(new Katz(alpha: $critical * 0.9)->isConvergent($star));
        self::assertFalse(new Katz(alpha: $critical * 1.1)->isConvergent($star));

        // And the predicate agrees with what of() actually does, which is the
        // only thing that makes it worth asking. Below the critical value the
        // scores come out finite; above it, of() refuses.
        $scores = new Katz(alpha: $critical * 0.9)->of($star);

        self::assertCount(5, $scores);
        self::assertGreaterThan($scores[1], $scores[0], 'the hub beats a leaf');

        foreach ($scores as $score) {
            self::assertTrue(is_finite($score));
        }

        $this->expectException(InvalidArgument::class);
        $this->expectExceptionMessageMatches('/must stay below/');
        new Katz(alpha: $critical * 1.1)->of($star);
    }

    /**
     * A graph with no edges has a spectral radius of zero, and every alpha
     * converges there -- the series is just beta at every node, with nothing
     * to propagate. Worth pinning because "no edges" reads like a degenerate
     * case that ought to be refused, and it is not one.
     */
    public function test_a_graph_with_no_edges_converges_for_any_alpha(): void
    {
        self::assertSame(0.0, Katz::spectralRadius(Graph::undirected(3)));
        self::assertSame(0.0, Katz::spectralRadius(Graph::undirected(0)));
        self::assertTrue(new Katz(alpha: 0.01)->isConvergent(Graph::undirected(3)));
        self::assertTrue(new Katz(alpha: 1000.0)->isConvergent(Graph::undirected(3)));

        // Every node scores beta and the vector is normalised, so all three
        // come out at 1/sqrt(3) -- equal, and not zero.
        self::assertEqualsWithDelta(
            [1.0 / sqrt(3.0), 1.0 / sqrt(3.0), 1.0 / sqrt(3.0)],
            new Katz(alpha: 0.5)->of(Graph::undirected(3)),
            1.0e-12,
        );
    }
}
