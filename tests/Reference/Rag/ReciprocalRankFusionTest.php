<?php

declare(strict_types=1);

namespace Vegoia\Tests\Reference\Rag;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Rag\NearestNeighbours;
use Vegoia\Rag\ReciprocalRankFusion;
use Vegoia\Tests\Support\Lre;
use Vegoia\Tests\Support\Paths;

/**
 * Rank fusion, against exact rational arithmetic.
 *
 * There is no library to compare with. RRF has no canonical implementation --
 * LangChain, Elasticsearch and Weaviate each read the conventions differently,
 * on whether ranks start at 0 or 1, on whether a document missing from a
 * ranking is charged a rank, on whether a duplicate within one ranking counts
 * twice. Generating from one of them would pin that library's reading rather
 * than the paper's, so the fixture computes the scores exactly with rationals
 * and rounds once.
 *
 * @see tools/generate_fusion_fixtures.py
 */
#[CoversClass(ReciprocalRankFusion::class)]
#[Group('reference')]
final class ReciprocalRankFusionTest extends TestCase
{
    /** @return iterable<string, array{string, array<string, mixed>}> */
    public static function cases(): iterable
    {
        /** @var array{cases: array<string, array<string, mixed>>} $decoded */
        $decoded = json_decode(
            (string) file_get_contents(Paths::fixture('rag/fusion.json')),
            true,
            flags: JSON_THROW_ON_ERROR,
        );

        foreach ($decoded['cases'] as $name => $entry) {
            yield $name => [$name, $entry];
        }
    }

    /** @param array<string, mixed> $entry */
    #[DataProvider('cases')]
    public function test_the_scores_match_exact_arithmetic(string $name, array $entry): void
    {
        /** @var array{rankings: list<list<string|int>>, k: int, fused: array{keys: list<string|int>, scores: list<float>}} $entry */
        $fused = ReciprocalRankFusion::fuse($entry['rankings'], $entry['k']);

        self::assertSame(
            $entry['fused']['keys'],
            array_keys($fused),
            "{$name}: the ranking or its order is wrong",
        );

        $index = 0;

        foreach ($fused as $score) {
            Lre::assertDigits(
                $score,
                $entry['fused']['scores'][$index],
                "{$name}: score at position {$index}",
                15.0,
            );
            $index++;
        }
    }

    /**
     * Scores never increase down the result, which is what makes it a ranking.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('cases')]
    public function test_the_result_is_in_descending_order(string $name, array $entry): void
    {
        /** @var array{rankings: list<list<string|int>>, k: int} $entry */
        $previous = INF;

        foreach (ReciprocalRankFusion::fuse($entry['rankings'], $entry['k']) as $key => $score) {
            self::assertLessThanOrEqual($previous, $score, "{$name}: at {$key}");
            $previous = $score;
        }
    }

    /**
     * Every key that appears anywhere is scored, and nothing else is.
     *
     * @param array<string, mixed> $entry
     */
    #[DataProvider('cases')]
    public function test_exactly_the_keys_that_appear_are_scored(string $name, array $entry): void
    {
        /** @var array{rankings: list<list<string|int>>, k: int} $entry */
        $seen = [];

        foreach ($entry['rankings'] as $ranking) {
            foreach ($ranking as $key) {
                $seen[$key] = true;
            }
        }

        $fused = ReciprocalRankFusion::fuse($entry['rankings'], $entry['k']);

        self::assertSame(count($seen), count($fused), "{$name}: wrong number of keys");

        foreach ($fused as $key => $score) {
            self::assertArrayHasKey($key, $seen, "{$name}: {$key} was invented");
            self::assertGreaterThan(0.0, $score, "{$name}: {$key} scored nothing");
        }
    }

    /**
     * The same multiset of ranks gives the same score, whatever order the
     * rankings listed them in.
     *
     * Floating addition is not associative, so without a canonical summation
     * order two documents appearing at the same set of positions can differ by
     * one unit in the last place -- and then the tie-break on the key never
     * fires and the ordering is decided by a rounding difference instead. The
     * fixture case built for this has 'a' at ranks 1, 2, 3 reached one way and
     * the comparison below reaches them the other.
     */
    public function test_the_score_depends_on_the_ranks_and_not_their_order(): void
    {
        // Ranks 1, 2 and 8, and the choice took three attempts to get right.
        // Of the 2024 triples of distinct ranks up to 24, 427 sum to different
        // doubles depending on the order -- but for most of those, including
        // {1, 2, 7}, it is only the middle-first orderings that differ, while
        // ascending and descending agree. Since this test reaches the ranks in
        // ascending order one way and descending the other, it needs a triple
        // where those two disagree, and {1, 2, 8} is the smallest:
        // 0.047228357233956519 against 0.047228357233956512.
        //
        // The first version used {1, 2, 3} and the second {1, 2, 7}, and both
        // passed with the canonical sort deleted.
        $eighth = ['p', 'q', 'r', 's', 't', 'u', 'v', 'a'];

        $forwards = ReciprocalRankFusion::fuse([['a'], ['x', 'a'], $eighth]);
        $backwards = ReciprocalRankFusion::fuse([$eighth, ['x', 'a'], ['a']]);

        self::assertSame(
            $forwards['a'],
            $backwards['a'],
            'the same ranks in a different order gave a different score, so the summation is '
            . 'not canonical and the tie-break on the key will never fire',
        );
    }

    /**
     * The point of the method, on the case built for it: a document nobody
     * ranks first can beat one that a single engine put at the top, because
     * three engines agreeing outweighs one being certain.
     */
    public function test_agreement_beats_a_single_enthusiastic_ranking(): void
    {
        $fused = ReciprocalRankFusion::fuse([
            ['x', 'b', 'c'],
            ['b', 'y', 'c'],
            ['b', 'c', 'z'],
        ]);

        self::assertSame('b', array_key_first($fused));
        self::assertGreaterThan($fused['x'], $fused['c'], 'never first, but always near the top');
    }

    /**
     * A ranking from NearestNeighbours drops straight in, which is the whole
     * point of the two agreeing on their tie-break.
     */
    public function test_it_composes_with_the_rest_of_the_namespace(): void
    {
        $corpus = ['a' => [1.0, 0.0], 'b' => [0.9, 0.1], 'c' => [0.0, 1.0]];

        $bySimilarity = array_keys(NearestNeighbours::cosine([1.0, 0.0], $corpus, 3));
        $byKeyword = ['c', 'b', 'a'];

        $fused = ReciprocalRankFusion::fuse([$bySimilarity, $byKeyword]);

        self::assertSame(['a', 'b', 'c'], $bySimilarity);
        self::assertCount(3, $fused);
        self::assertSame(['a', 'c', 'b'], array_keys($fused));
    }

    /**
     * Being first somewhere beats being second everywhere, and that surprises
     * people, so it is pinned rather than left to be discovered.
     *
     * With two rankings that reverse each other, a document at positions 1 and
     * 3 scores 1/61 + 1/63 = 124/3843, while one at 2 and 2 scores 2/62 =
     * 1/31, and the first is larger. It falls out of 1/(k + r) being convex:
     * the gain from moving up to first outweighs the loss from dropping to
     * third.
     *
     * This does not contradict what the method is for. Consensus wins by
     * appearing in more of the rankings, which is the case above; among
     * documents that appear in all of them, the extremes win.
     */
    public function test_first_and_third_beats_second_and_second(): void
    {
        $fused = ReciprocalRankFusion::fuse([['a', 'b', 'c'], ['c', 'b', 'a']]);

        self::assertSame(['a', 'c', 'b'], array_keys($fused));
        self::assertSame($fused['a'], $fused['c'], 'the two extremes are symmetric');
        self::assertGreaterThan($fused['b'], $fused['a']);

        // 124/3843 against 1/31, checked as the exact rationals they are.
        self::assertEqualsWithDelta(124.0 / 3843.0, $fused['a'], 1.0e-15);
        self::assertEqualsWithDelta(1.0 / 31.0, $fused['b'], 1.0e-15);
    }

    public function test_no_rankings_gives_no_result(): void
    {
        self::assertSame([], ReciprocalRankFusion::fuse([]));
        self::assertSame([], ReciprocalRankFusion::fuse([[], []]));
    }

    public function test_a_negative_constant_is_refused(): void
    {
        $this->expectException(InvalidArgument::class);
        $this->expectExceptionMessageMatches('/Rank fusion constant/');

        ReciprocalRankFusion::fuse([['a']], -1);
    }

    /** Zero is legitimate -- the pure reciprocal rank -- and must be accepted. */
    public function test_a_constant_of_zero_is_allowed(): void
    {
        self::assertSame(1.0, ReciprocalRankFusion::fuse([['a']], 0)['a']);
    }
}
