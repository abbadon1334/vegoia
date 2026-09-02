<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit\Rag;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Rag\MaximalMarginalRelevance;
use Vegoia\Rag\Similarity;

/**
 * A vector is its values, not its keys.
 *
 * These routines walk two vectors together by position, reading the second
 * with the first one's keys. That is correct for a list and silently wrong for
 * anything else: an array keyed 5, 9, 12 -- which is what array_filter leaves
 * behind, and what a caller who dropped a dimension has -- makes every lookup
 * miss. PHP answers a missing key with null and a warning, so the sum runs to
 * completion and returns a number. In production, with warnings off, that
 * number is the answer.
 *
 * This is the same defect that was found and fixed in Correlation, where a
 * Pearson coefficient came back 0.596 instead of 0.965 for the same reason.
 * Fixed the same way, so the two halves of the library agree on what a vector
 * is.
 */
#[CoversClass(Similarity::class)]
#[CoversClass(MaximalMarginalRelevance::class)]
final class VectorKeyingTest extends TestCase
{
    /**
     * The same three numbers under every keying anyone might arrive with.
     *
     * @return iterable<string, array{array<array-key, float>, array<array-key, float>}>
     */
    public static function keyings(): iterable
    {
        yield 'a list, as documented' => [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // What array_filter leaves behind after dropping a dimension.
        yield 'gaps on the left' => [[5 => 1.0, 9 => 2.0, 12 => 3.0], [4.0, 5.0, 6.0]];
        yield 'gaps on the right' => [[1.0, 2.0, 3.0], [5 => 4.0, 9 => 5.0, 12 => 6.0]];
        yield 'gaps on both' => [[5 => 1.0, 9 => 2.0, 12 => 3.0], [2 => 4.0, 7 => 5.0, 8 => 6.0]];

        // A named embedding, which is how a caller keeps track of dimensions.
        yield 'string keys' => [['x' => 1.0, 'y' => 2.0, 'z' => 3.0], ['x' => 4.0, 'y' => 5.0, 'z' => 6.0]];

        // Same values in the same order, keys counting down. array_reverse
        // with preserve_keys, and the one case that already worked.
        yield 'descending integer keys' => [[2 => 1.0, 1 => 2.0, 0 => 3.0], [4.0, 5.0, 6.0]];
    }

    /**
     * Reference values from numpy on the same three-by-three numbers:
     * a.b = 32, |a-b| = 5.196152422706632, cos = 0.974631846197076.
     *
     * @param array<array-key, float> $a
     * @param array<array-key, float> $b
     */
    #[DataProvider('keyings')]
    public function test_every_keying_of_the_same_vector_scores_the_same(array $a, array $b): void
    {
        self::assertEqualsWithDelta(32.0, Similarity::dot($a, $b), 1.0e-13);
        self::assertEqualsWithDelta(5.196152422706632, Similarity::euclidean($a, $b), 1.0e-13);
        self::assertEqualsWithDelta(0.974631846197076, Similarity::cosine($a, $b), 1.0e-13);
    }

    /**
     * The re-ranker inherits the fix, since it reaches the similarity through
     * the same door -- and it is the more likely place for an oddly-keyed
     * vector to arrive, because candidates come from whatever a vector store
     * handed back.
     */
    public function test_the_re_ranker_is_indifferent_to_how_vectors_are_keyed(): void
    {
        $query = [1.0, 0.0, 0.0];

        $plain = [
            'near' => [0.9, 0.1, 0.0],
            'twin' => [0.9, 0.1, 0.0],
            'far' => [0.0, 0.0, 1.0],
        ];

        $awkward = [
            'near' => [3 => 0.9, 8 => 0.1, 11 => 0.0],
            'twin' => ['a' => 0.9, 'b' => 0.1, 'c' => 0.0],
            'far' => [2 => 0.0, 1 => 0.0, 0 => 1.0],
        ];

        self::assertSame(
            MaximalMarginalRelevance::select($query, $plain, 3, 0.5),
            MaximalMarginalRelevance::select($query, $awkward, 3, 0.5),
        );

        // And the answer is the one MMR exists to give: the duplicate is
        // pushed behind the unrelated document.
        self::assertSame(['near', 'far', 'twin'], MaximalMarginalRelevance::select($query, $plain, 3, 0.5));
    }

    /** A mismatch in length is still a mismatch, whatever the keys look like. */
    public function test_vectors_of_different_lengths_are_still_refused(): void
    {
        $this->expectException(InvalidArgument::class);

        Similarity::dot([5 => 1.0, 9 => 2.0], [1.0, 2.0, 3.0]);
    }
}
