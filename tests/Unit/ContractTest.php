<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit;

use Generator;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Graph\Centrality\Katz;
use Vegoia\Graph\Centrality\PageRank;
use Vegoia\Graph\Community\Agreement;
use Vegoia\Graph\Community\Leiden;
use Vegoia\Graph\Community\Quality\Modularity;
use Vegoia\Graph\Graph;
use Vegoia\Graph\Partition;
use Vegoia\Graph\Path\BreadthFirst;
use Vegoia\Graph\Path\Dijkstra;
use Vegoia\Rag\MaximalMarginalRelevance;
use Vegoia\Rag\NearestNeighbours;
use Vegoia\Stats\Correlation;
use Vegoia\Stats\Descriptive;
use Vegoia\Stats\OneWayAnova;
use Vegoia\Stats\Regression\LeastSquares;

/**
 * Every argument the library refuses, and the nearest one it accepts.
 *
 * The pairing is the point. A test that only checks the rejection pins the
 * existence of a guard but not where it sits: `$k < 1` and `$k < 2` both throw
 * on zero, and only one of them is right. Checking the value just inside the
 * boundary as well is what distinguishes them, and it is what mutation testing
 * looks for -- an off-by-one in a guard is the single most common surviving
 * mutant in code like this, and until this file existed 79 of them survived
 * here.
 *
 * Written from the throw sites rather than from memory: every
 * `throw InvalidArgument::` in src has a row below, so a new guard without a
 * test is visible as an absence rather than hidden as a pass.
 */
final class ContractTest extends TestCase
{
    /**
     * Arguments that must be refused.
     *
     * @return iterable<string, array{callable(): mixed}>
     */
    public static function refused(): iterable
    {
        $graph = Graph::undirected(3, [[0, 1], [1, 2]]);
        $directed = Graph::directed(3, [[0, 1], [1, 2]]);

        yield 'a graph of negative order' => [static fn () => Graph::undirected(-1)];
        // The declared type forbids both of these, which is why the runtime
        // guard is worth having and worth testing: the type is advice to a
        // static analyser, and a caller without one still gets an answer.
        /** @phpstan-ignore argument.type */
        yield 'an edge that is not a pair' => [static fn () => Graph::undirected(2, [[0]])];
        /** @phpstan-ignore argument.type */
        yield 'an edge that is not an array' => [static fn () => Graph::undirected(2, ['nonsense'])];
        yield 'an edge leaving the node set' => [static fn () => Graph::undirected(2, [[0, 5]])];
        yield 'an edge arriving from outside' => [static fn () => Graph::undirected(2, [[-1, 0]])];
        yield 'an infinite edge weight' => [static fn () => Graph::undirected(2, [[0, 1, INF]])];
        yield 'a degree outside the node set' => [static fn () => $graph->degree(3)];
        yield 'neighbours outside the node set' => [static fn () => iterator_to_array($graph->neighbours(3))];

        yield 'a negative resolution' => [static fn () => new Modularity(-1.0)];
        yield 'zero randomness in Leiden' => [static fn () => new Leiden(new Modularity(), 1, 0.0)];
        yield 'zero maximum iterations' => [static fn () => new Leiden(new Modularity(), 1, 0.01, 0)];
        yield 'zero passes' => [static fn () => new Leiden(new Modularity(), 1, 0.01, 10, 0)];
        yield 'Leiden on a directed graph' => [
            static fn () => Leiden::modularity(seed: 1)->partition($directed),
        ];

        yield 'a damping factor above one' => [static fn () => new PageRank(1.5)];
        yield 'a negative damping factor' => [static fn () => new PageRank(-0.1)];
        // The bounds are open: a damping factor of exactly 1 never restarts
        // and exactly 0 never follows a link, and neither is PageRank.
        yield 'a damping factor of exactly one' => [static fn () => new PageRank(1.0)];
        yield 'a damping factor of exactly zero' => [static fn () => new PageRank(0.0)];
        yield 'a zero Katz alpha' => [static fn () => new Katz(0.0)];

        yield 'a breadth-first source outside the graph' => [
            static fn () => BreadthFirst::distancesFrom($graph, 3),
        ];
        yield 'a Dijkstra source outside the graph' => [static fn () => Dijkstra::distancesFrom($graph, 3)];
        yield 'a Dijkstra destination outside the graph' => [static fn () => Dijkstra::shortestPath($graph, 0, 3)];
        yield 'a negative edge weight in Dijkstra' => [
            static fn () => Dijkstra::distancesFrom(Graph::undirected(2, [[0, 1, -1.0]]), 0),
        ];

        yield 'agreement between partitions of different sizes' => [
            static fn () => Agreement::adjustedRandIndex(
                Partition::fromMembership([0, 0, 1]),
                Partition::fromMembership([0, 1]),
            ),
        ];

        yield 'the mean of nothing' => [static fn () => Descriptive::of([])->mean()];
        yield 'the variance of nothing' => [static fn () => Descriptive::of([])->variance()];
        yield 'a correlation of one point' => [static fn () => Correlation::pearson([1.0], [2.0])];
        yield 'a correlation of unequal lengths' => [
            static fn () => Correlation::pearson([1.0, 2.0], [1.0, 2.0, 3.0]),
        ];
        yield 'a Spearman of unequal lengths' => [
            static fn () => Correlation::spearman([1.0, 2.0], [1.0]),
        ];
        yield 'a Kendall of unequal lengths' => [
            static fn () => Correlation::kendall([1.0, 2.0], [1.0]),
        ];

        yield 'an analysis of variance of one group' => [static fn () => OneWayAnova::of([[1.0, 2.0]])];
        yield 'an analysis of variance with an empty group' => [
            static fn () => OneWayAnova::of([[1.0, 2.0], []]),
        ];
        yield 'grouped values against mismatched labels' => [
            static fn () => OneWayAnova::grouped([1.0, 2.0], ['a']),
        ];

        yield 'a regression with no data' => [static fn () => LeastSquares::fit([], [])];
        yield 'a regression with mismatched lengths' => [
            static fn () => LeastSquares::fit([[1.0], [2.0]], [1.0]),
        ];
        yield 'a regression with a ragged design' => [
            static fn () => LeastSquares::fit([[1.0, 2.0], [3.0]], [1.0, 2.0]),
        ];
        yield 'a regression with fewer rows than parameters' => [
            static fn () => LeastSquares::fit([[1.0, 2.0]], [1.0]),
        ];
        yield 'a polynomial of degree zero' => [
            static fn () => LeastSquares::polynomial([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 0),
        ];
        yield 'a polynomial with no data' => [static fn () => LeastSquares::polynomial([], [], 1)];
        yield 'a polynomial with mismatched lengths' => [
            static fn () => LeastSquares::polynomial([1.0, 2.0], [1.0], 1),
        ];

        yield 'zero neighbours' => [
            static fn () => NearestNeighbours::cosine([1.0], ['a' => [1.0]], 0),
        ];
        yield 'zero selections' => [
            static fn () => MaximalMarginalRelevance::select([1.0], ['a' => [1.0]], 0, 0.5),
        ];
        yield 'a lambda above one' => [
            static fn () => MaximalMarginalRelevance::select([1.0], ['a' => [1.0]], 1, 1.5),
        ];
        yield 'a negative lambda' => [
            static fn () => MaximalMarginalRelevance::select([1.0], ['a' => [1.0]], 1, -0.5),
        ];
    }

    /** @param callable(): mixed $call */
    #[DataProvider('refused')]
    public function test_it_refuses_what_it_cannot_answer(callable $call): void
    {
        $this->expectException(InvalidArgument::class);

        $call();
    }

    /**
     * The nearest argument on the other side of each boundary, which must be
     * accepted. Without these, a guard could sit one place too far in and
     * every rejection test above would still pass.
     *
     * @return iterable<string, array{callable(): mixed}>
     */
    public static function accepted(): iterable
    {
        yield 'a graph of order zero' => [static fn () => Graph::undirected(0)];
        yield 'the last node in the set' => [static fn () => Graph::undirected(3, [[0, 2]])->degree(2)];
        yield 'a resolution of zero' => [static fn () => new Modularity(0.0)];
        yield 'the smallest usable randomness' => [
            static fn () => new Leiden(new Modularity(), 1, PHP_FLOAT_EPSILON),
        ];
        yield 'a single maximum iteration' => [static fn () => new Leiden(new Modularity(), 1, 0.01, 1)];
        yield 'a single pass' => [static fn () => new Leiden(new Modularity(), 1, 0.01, 10, 1)];
        yield 'a damping factor just inside one' => [static fn () => new PageRank(0.999999)];
        yield 'a damping factor just above zero' => [static fn () => new PageRank(1.0e-9)];
        yield 'the smallest usable Katz alpha' => [static fn () => new Katz(PHP_FLOAT_EPSILON)];

        yield 'the mean of one value' => [static fn () => Descriptive::of([1.0])->mean()];
        yield 'a correlation of two points' => [
            static fn () => Correlation::pearson([1.0, 2.0], [2.0, 1.0]),
        ];
        yield 'an analysis of variance of two groups' => [
            static fn () => OneWayAnova::of([[1.0, 2.0], [3.0, 4.0]]),
        ];
        yield 'a group of one observation' => [
            static fn () => OneWayAnova::of([[1.0], [3.0, 4.0]]),
        ];
        yield 'a regression with exactly as many rows as parameters' => [
            static fn () => LeastSquares::fit([[1.0], [2.0]], [1.0, 2.0]),
        ];
        yield 'a polynomial of degree one' => [
            static fn () => LeastSquares::polynomial([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1),
        ];
        yield 'one neighbour' => [
            static fn () => NearestNeighbours::cosine([1.0], ['a' => [1.0]], 1),
        ];
        yield 'a lambda of one' => [
            static fn () => MaximalMarginalRelevance::select([1.0], ['a' => [1.0]], 1, 1.0),
        ];
        yield 'a lambda of zero' => [
            static fn () => MaximalMarginalRelevance::select([1.0], ['a' => [1.0]], 1, 0.0),
        ];
    }

    /** @param callable(): mixed $call */
    #[DataProvider('accepted')]
    public function test_it_accepts_the_value_just_inside_each_boundary(callable $call): void
    {
        $call();

        // Reaching here without an exception is the assertion; PHPUnit is told
        // so explicitly rather than being left to call the test risky.
        $this->expectNotToPerformAssertions();
    }

    /**
     * Input arrives as a list, whatever shape it was handed in.
     *
     * A Generator yields its own keys, an array may be keyed by anything, and
     * every numeric routine downstream indexes by position. Normalising is a
     * step that does nothing visible on the inputs anybody writes in a test,
     * which is exactly why nothing noticed it could be removed.
     */
    public function test_it_normalises_however_the_values_arrive(): void
    {
        $expected = Descriptive::of([1.0, 2.0, 3.0, 4.0])->mean();

        $keyed = [7 => 1.0, 'x' => 2.0, 3 => 3.0, 'y' => 4.0];
        self::assertSame($expected, Descriptive::of($keyed)->mean(), 'an array keyed by anything');

        $generator = (static function (): Generator {
            yield 'a' => 1.0;
            yield 'b' => 2.0;
            yield 'c' => 3.0;
            yield 'd' => 4.0;
        })();

        self::assertSame($expected, Descriptive::of($generator)->mean(), 'a generator with string keys');

        $repeating = (static function (): Generator {
            yield 0 => 1.0;
            yield 0 => 2.0;
            yield 0 => 3.0;
            yield 0 => 4.0;
        })();

        // The one that actually bites: a generator may yield the same key
        // repeatedly, and collecting it with the keys preserved would keep one
        // value out of four.
        self::assertSame($expected, Descriptive::of($repeating)->mean(), 'a generator repeating a key');
    }

    /** The same normalisation, on the two-sample routines. */
    public function test_correlation_normalises_however_the_values_arrive(): void
    {
        $expected = Correlation::pearson([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 5.0, 9.0]);

        self::assertSame(
            $expected,
            Correlation::pearson(
                [5 => 1.0, 'a' => 2.0, 1 => 3.0, 'b' => 4.0],
                [9 => 2.0, 'c' => 4.0, 0 => 5.0, 'd' => 9.0],
            ),
        );
    }
}
