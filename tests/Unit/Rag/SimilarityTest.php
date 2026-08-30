<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit\Rag;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\TestCase;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Rag\Similarity;

#[CoversClass(Similarity::class)]
final class SimilarityTest extends TestCase
{
    public function test_cosine_of_identical_directions_is_one_regardless_of_length(): void
    {
        self::assertSame(1.0, Similarity::cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]));
        self::assertSame(1.0, Similarity::cosine([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]));
    }

    public function test_cosine_of_orthogonal_vectors_is_zero_and_of_opposite_is_minus_one(): void
    {
        self::assertSame(0.0, Similarity::cosine([1.0, 0.0], [0.0, 1.0]));
        self::assertSame(-1.0, Similarity::cosine([1.0, 0.0], [-1.0, 0.0]));
    }

    /**
     * A zero vector has no direction, so no angle to anything. Returning 0.0
     * would claim orthogonality, which is a statement the data does not support
     * and which quietly ranks an empty embedding above a genuinely dissimilar
     * one in a retrieval set.
     */
    public function test_cosine_against_a_zero_vector_is_refused_rather_than_guessed(): void
    {
        $this->expectException(InvalidArgument::class);

        Similarity::cosine([0.0, 0.0], [1.0, 1.0]);
    }

    public function test_it_refuses_vectors_of_different_lengths(): void
    {
        $this->expectException(InvalidArgument::class);

        Similarity::cosine([1.0, 2.0], [1.0, 2.0, 3.0]);
    }

    public function test_euclidean_distance_is_the_hypotenuse(): void
    {
        self::assertSame(5.0, Similarity::euclidean([0.0, 0.0], [3.0, 4.0]));
        self::assertSame(0.0, Similarity::euclidean([1.5, -2.0], [1.5, -2.0]));
    }

    public function test_dot_product_of_a_known_pair(): void
    {
        self::assertSame(32.0, Similarity::dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]));
    }

    public function test_jaccard_is_the_overlap_over_the_union(): void
    {
        // {a,b} and {b,c} share one element out of three distinct ones.
        self::assertEqualsWithDelta(1 / 3, Similarity::jaccard(['a', 'b'], ['b', 'c']), 1.0e-15);
        self::assertSame(0.5, Similarity::jaccard(['a', 'b'], ['b']));
        self::assertSame(1.0, Similarity::jaccard(['a', 'b'], ['b', 'a']));
        self::assertSame(0.0, Similarity::jaccard(['a'], ['b']));
    }

    /**
     * Two empty sets differ in nothing. Returning 0.0 would call them
     * completely dissimilar, which reads as a real change when comparing
     * successive snapshots that both happen to be empty.
     */
    public function test_two_empty_sets_are_identical_not_dissimilar(): void
    {
        self::assertSame(1.0, Similarity::jaccard([], []));
        self::assertSame(0.0, Similarity::jaccard([], ['a']));
    }

    public function test_jaccard_ignores_order_and_duplicates(): void
    {
        self::assertSame(1.0, Similarity::jaccard(['a', 'a', 'b'], ['b', 'a']));
    }

    public function test_cosine_matches_the_hand_computed_value(): void
    {
        // (1*4 + 2*5) / (sqrt(5) * sqrt(41)) = 14 / sqrt(205)
        self::assertEqualsWithDelta(
            14.0 / sqrt(205.0),
            Similarity::cosine([1.0, 2.0], [4.0, 5.0]),
            1.0e-15,
        );
    }
}
