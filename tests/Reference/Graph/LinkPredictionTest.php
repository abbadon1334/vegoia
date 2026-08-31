<?php

declare(strict_types=1);

namespace Vegoia\Tests\Reference\Graph;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Vegoia\Graph\Graph;
use Vegoia\Graph\LinkMeasure;
use Vegoia\Graph\LinkPrediction;
use Vegoia\Tests\Support\GraphFixture;
use Vegoia\Tests\Support\Lre;
use Vegoia\Tests\Support\Paths;

/**
 * Link prediction, against NetworkX on every pair of nine graphs.
 *
 * All five measures are checked on all 1511 unordered pairs, adjacent ones
 * included. NetworkX defines them on any pair -- adjacency has nothing to do
 * with the arithmetic -- and testing only the non-adjacent ones would leave
 * an implementation free to assume something it was never told.
 *
 * @see tools/generate_link_prediction_fixtures.py
 */
#[CoversClass(LinkPrediction::class)]
#[CoversClass(LinkMeasure::class)]
#[Group('reference')]
final class LinkPredictionTest extends TestCase
{
    /** @var array<string, mixed>|null */
    private static ?array $fixture = null;

    /** @return array<string, mixed> */
    private static function graphs(): array
    {
        if (self::$fixture === null) {
            /** @var array{graphs: array<string, mixed>} $decoded */
            $decoded = json_decode(
                (string) file_get_contents(Paths::fixture('link_prediction.json')),
                true,
                flags: JSON_THROW_ON_ERROR,
            );
            self::$fixture = $decoded['graphs'];
        }

        return self::$fixture;
    }

    /** @return iterable<string, array{string, array<string, mixed>}> */
    public static function graphsWithMeasures(): iterable
    {
        /** @var array<string, array<string, mixed>> $graphs */
        $graphs = self::graphs();

        foreach ($graphs as $name => $entry) {
            yield $name => [$name, $entry];
        }
    }

    /** @param array<string, mixed> $entry */
    #[DataProvider('graphsWithMeasures')]
    public function test_every_measure_matches_networkx_on_every_pair(string $name, array $entry): void
    {
        /** @var array{measures: array<string, array<string, float|int>>} $entry */
        $graph = GraphFixture::load($name)->graph();

        $measures = [
            'common_neighbours' => LinkMeasure::CommonNeighbours,
            'jaccard' => LinkMeasure::Jaccard,
            'adamic_adar' => LinkMeasure::AdamicAdar,
            'resource_allocation' => LinkMeasure::ResourceAllocation,
            'preferential_attachment' => LinkMeasure::PreferentialAttachment,
        ];

        foreach ($measures as $key => $measure) {
            foreach ($entry['measures'][$key] as $pair => $expected) {
                [$u, $v] = array_map(intval(...), explode(',', $pair));

                Lre::assertDigits(
                    LinkPrediction::score($graph, $u, $v, $measure),
                    (float) $expected,
                    "{$name}: {$key} of ({$u}, {$v})",
                    14.0,
                );
            }
        }
    }

    /**
     * Every measure is symmetric on an undirected graph.
     *
     * Cheap to state and easy to get wrong: three of the five walk one
     * neighbourhood and look up the other, so a mistake in which is which is
     * invisible until the pair is lopsided.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('graphsWithMeasures')]
    public function test_the_measures_do_not_depend_on_the_order_of_the_pair(
        string $name,
        array $entry,
    ): void {
        $graph = GraphFixture::load($name)->graph();
        $order = $graph->order();

        foreach (LinkMeasure::cases() as $measure) {
            for ($u = 0; $u < $order; $u++) {
                for ($v = $u + 1; $v < $order; $v++) {
                    self::assertSame(
                        LinkPrediction::score($graph, $u, $v, $measure),
                        LinkPrediction::score($graph, $v, $u, $measure),
                        "{$name}: {$measure->name} is not symmetric at ({$u}, {$v})",
                    );
                }
            }
        }
    }

    /**
     * The ranking returns the same scores the pairwise call does, in order,
     * and only over nodes that are not already neighbours.
     *
     * This is the entry point a retrieval system actually uses -- "which
     * entities is this one probably related to" -- and it takes a different
     * route to the answer, walking two hops rather than scoring every pair,
     * so agreeing with the pairwise call is a real check rather than a
     * tautology.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('graphsWithMeasures')]
    public function test_the_ranking_agrees_with_the_pairwise_scores(string $name, array $entry): void
    {
        $graph = GraphFixture::load($name)->graph();
        $ranked = [];

        foreach (LinkMeasure::cases() as $measure) {
            for ($node = 0; $node < $graph->order(); $node++) {
                $ranked = LinkPrediction::rank($graph, $node, $measure);
                $previous = INF;

                foreach ($ranked as $other => $score) {
                    self::assertNotSame($node, $other, "{$name}: a node was ranked against itself");
                    self::assertFalse(
                        $graph->hasEdge($node, $other),
                        "{$name}: {$other} is already a neighbour of {$node}",
                    );

                    self::assertSame(
                        LinkPrediction::score($graph, $node, $other, $measure),
                        $score,
                        "{$name}: ranking and pairwise disagree for ({$node}, {$other})",
                    );

                    self::assertLessThanOrEqual(
                        $previous,
                        $score,
                        "{$name}: the ranking is not in descending order",
                    );
                    $previous = $score;
                }
            }
        }

        // A complete graph has no missing edges, so every ranking above is
        // empty and the loop asserts nothing. Saying so out loud is the
        // difference between a test that found nothing and a test that ran
        // nothing.
        if ($graph->size() === $graph->order() * ($graph->order() - 1) / 2) {
            self::assertSame([], $ranked, "{$name}: a complete graph can have nothing predicted");
        }
    }

    /**
     * Preferential attachment ignores shared neighbours, so it ranks every
     * candidate a node has -- the other four cannot score a pair with nothing
     * in common above zero, and a zero is not a prediction.
     */
    public function test_the_ranking_omits_candidates_with_no_evidence(): void
    {
        // Two triangles joined by nothing: from inside one of them, nobody in
        // the other shares a neighbour.
        $graph = Graph::undirected(6, [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]]);

        foreach ([LinkMeasure::CommonNeighbours, LinkMeasure::Jaccard,
                  LinkMeasure::AdamicAdar, LinkMeasure::ResourceAllocation] as $measure) {
            self::assertSame(
                [],
                LinkPrediction::rank($graph, 0, $measure),
                "{$measure->name} invented a candidate across a disconnected component",
            );
        }

        self::assertNotSame(
            [],
            LinkPrediction::rank($graph, 0, LinkMeasure::PreferentialAttachment),
            'preferential attachment scores by degree alone, so it always has candidates',
        );
    }

    /** A limit truncates the ranking without reordering it. */
    public function test_a_limit_takes_the_top_of_the_same_ranking(): void
    {
        $graph = GraphFixture::load('zachary')->graph();
        $full = LinkPrediction::rank($graph, 0, LinkMeasure::AdamicAdar);

        self::assertGreaterThan(3, count($full));
        self::assertSame(array_slice($full, 0, 3, preserve_keys: true), LinkPrediction::rank($graph, 0, LinkMeasure::AdamicAdar, 3));
    }
}
