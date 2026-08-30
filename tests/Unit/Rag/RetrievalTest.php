<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit\Rag;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\TestCase;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Rag\MaximalMarginalRelevance;
use Vegoia\Rag\NearestNeighbours;

#[CoversClass(NearestNeighbours::class)]
#[CoversClass(MaximalMarginalRelevance::class)]
final class RetrievalTest extends TestCase
{
    /** @var array<string, list<float>> */
    private const array CORPUS = [
        'north' => [0.0, 1.0],
        'north-ish' => [0.1, 1.0],
        'east' => [1.0, 0.0],
        'south' => [0.0, -1.0],
    ];

    /**
     * A corpus where diversity actually costs relevance.
     *
     * The obvious test corpus does not discriminate: when the query equals one
     * of the vectors, the first pick is that vector, and every remaining
     * candidate d then has sim(d, picked) == sim(d, query). The MMR score
     * collapses to (2*lambda - 1) * sim, so at lambda 0.5 every candidate
     * scores exactly zero and the winner is decided by iteration order rather
     * than by the algorithm. Here the query sits *between* the near-duplicates
     * instead, so the two terms differ.
     *
     * @var list<float>
     */
    private const array QUERY = [1.0, 0.1];

    /** @var array<string, list<float>> */
    private const array SPREAD = [
        'east' => [1.0, 0.0],        // relevance 0.99504
        'east-ish' => [1.0, 0.05],   // relevance 0.99875, and near-identical to east
        'north' => [0.0, 1.0],       // relevance 0.09950, but unlike either
    ];

    public function test_it_returns_the_closest_vectors_in_order(): void
    {
        $hits = NearestNeighbours::cosine([0.0, 1.0], self::CORPUS, 2);

        self::assertSame(['north', 'north-ish'], array_keys($hits));
        self::assertEqualsWithDelta(1.0, $hits['north'], 1.0e-15);
    }

    public function test_asking_for_more_than_exists_returns_everything(): void
    {
        self::assertCount(4, NearestNeighbours::cosine([0.0, 1.0], self::CORPUS, 99));
    }

    public function test_ties_are_broken_by_key_so_results_are_reproducible(): void
    {
        $corpus = ['b' => [1.0, 0.0], 'a' => [1.0, 0.0], 'c' => [1.0, 0.0]];

        self::assertSame(['a', 'b'], array_keys(NearestNeighbours::cosine([1.0, 0.0], $corpus, 2)));
    }

    public function test_it_refuses_a_non_positive_k(): void
    {
        $this->expectException(InvalidArgument::class);

        NearestNeighbours::cosine([0.0, 1.0], self::CORPUS, 0);
    }

    /**
     * Pure relevance ranking returns near-duplicates: 'north' and 'north-ish'
     * say the same thing, and spending two of three slots on them wastes the
     * context window. With lambda below 1, MMR trades a little relevance for
     * coverage and picks something different second.
     */
    public function test_it_prefers_a_diverse_second_result_over_a_near_duplicate(): void
    {
        $relevanceOnly = MaximalMarginalRelevance::select(self::QUERY, self::SPREAD, 2, lambda: 1.0);
        self::assertSame(['east-ish', 'east'], $relevanceOnly, 'pure relevance takes the duplicate');

        $diverse = MaximalMarginalRelevance::select(self::QUERY, self::SPREAD, 2, lambda: 0.5);
        self::assertSame(['east-ish', 'north'], $diverse, 'balanced selection skips the duplicate');
    }

    /**
     * Nothing has been selected when the first pick is made, so the redundancy
     * term is vacuous and lambda cannot express a preference. Leaving that to
     * iteration order would make `lambda: 0.0` return an arbitrary document as
     * its top result, so the first pick is defined to be the most relevant one
     * at every lambda.
     */
    public function test_the_first_pick_is_always_the_most_relevant(): void
    {
        foreach ([0.0, 0.3, 0.7, 1.0] as $lambda) {
            $selected = MaximalMarginalRelevance::select([1.0, 0.0], self::CORPUS, 3, $lambda);

            self::assertSame('east', $selected[0], "lambda {$lambda}");
        }
    }

    public function test_lambda_must_lie_between_zero_and_one(): void
    {
        $this->expectException(InvalidArgument::class);

        MaximalMarginalRelevance::select([0.0, 1.0], self::CORPUS, 2, lambda: 1.5);
    }

    public function test_an_empty_corpus_yields_nothing(): void
    {
        self::assertSame([], NearestNeighbours::cosine([1.0, 0.0], [], 3));
        self::assertSame([], MaximalMarginalRelevance::select([1.0, 0.0], [], 3, 0.5));
    }
}
