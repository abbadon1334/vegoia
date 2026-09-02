<?php

declare(strict_types=1);

namespace Vegoia\Rag;

use function array_flip;
use function array_values;
use function count;
use function max;
use function min;
use function sqrt;

use Vegoia\Exception\InvalidArgument;
use Vegoia\Support\CompensatedSum;

/**
 * Similarity measures for retrieval.
 *
 * Degenerate inputs are refused rather than given a plausible number. A zero
 * vector has no direction, so its cosine to anything is undefined -- returning
 * 0.0 would assert orthogonality and quietly rank an empty embedding above a
 * genuinely unrelated document. Failing loudly at the point of the bad vector
 * is cheaper than explaining an odd retrieval result later.
 */
final class Similarity
{
    /**
     * Keys are ignored and the values read in order -- see assertSameShape.
     *
     * @param array<array-key, float> $a
     * @param array<array-key, float> $b
     */
    public static function cosine(array $a, array $b): float
    {
        [$a, $b] = self::assertSameShape($a, $b);

        $dot = new CompensatedSum();
        $normA = new CompensatedSum();
        $normB = new CompensatedSum();

        foreach ($a as $i => $value) {
            $other = $b[$i];
            $dot->add($value * $other);
            $normA->add($value * $value);
            $normB->add($other * $other);
        }

        $magnitude = sqrt($normA->value()) * sqrt($normB->value());

        if ($magnitude === 0.0) {
            throw InvalidArgument::malformedEdge(
                'Cosine similarity is undefined against a zero vector'
            );
        }

        // Rounding can push an identical pair a hair past 1.0, which then
        // produces NaN in any caller taking acos() of the result.
        return max(-1.0, min(1.0, $dot->value() / $magnitude));
    }

    /**
     * Keys are ignored and the values read in order -- see assertSameShape.
     *
     * @param array<array-key, float> $a
     * @param array<array-key, float> $b
     */
    public static function dot(array $a, array $b): float
    {
        [$a, $b] = self::assertSameShape($a, $b);

        $sum = new CompensatedSum();

        foreach ($a as $i => $value) {
            $sum->add($value * $b[$i]);
        }

        return $sum->value();
    }

    /**
     * Keys are ignored and the values read in order -- see assertSameShape.
     *
     * @param array<array-key, float> $a
     * @param array<array-key, float> $b
     */
    public static function euclidean(array $a, array $b): float
    {
        [$a, $b] = self::assertSameShape($a, $b);

        $sum = new CompensatedSum();

        foreach ($a as $i => $value) {
            $sum->add(($value - $b[$i]) ** 2);
        }

        return sqrt($sum->value());
    }

    /**
     * Overlap over union, on sets of anything scalar. Duplicates and order are
     * ignored, because a set has neither.
     *
     * Two empty sets score 1.0: they differ in nothing. Scoring them 0.0 makes
     * a comparison of two empty snapshots look like a total change, which is
     * exactly backwards.
     *
     * @param list<string|int> $a
     * @param list<string|int> $b
     */
    public static function jaccard(array $a, array $b): float
    {
        $left = array_flip($a);
        $right = array_flip($b);

        if ($left === [] && $right === []) {
            return 1.0;
        }

        $shared = 0;

        foreach ($left as $key => $_) {
            if (isset($right[$key])) {
                $shared++;
            }
        }

        $union = count($left) + count($right) - $shared;

        return $union === 0 ? 1.0 : $shared / $union;
    }

    /**
     * Keys are ignored and the values read in order -- see assertSameShape.
     *
     * @param array<array-key, float> $a
     * @param array<array-key, float> $b
     */
    /**
     * Check the shape, and hand back both vectors indexed by position.
     *
     * The re-indexing is not tidiness. These routines walk the two vectors
     * together, reading the second with the first one's keys, which is right
     * for a list and silently wrong for anything else. An array left keyed
     * 5, 9, 12 -- what array_filter returns after dropping a dimension --
     * makes every lookup miss, and PHP answers a missing key with null and a
     * warning rather than an error, so the sum completes and returns a number.
     * Worse, keys that all exist in the wrong order raise nothing at all:
     * [2 => 1.0, 1 => 2.0, 0 => 3.0] against [4.0, 5.0, 6.0] gave a dot
     * product of 28 where the answer is 32, with no warning to notice.
     *
     * The declared type is list<float>, so position is the whole meaning of a
     * vector here and keys carry none. Correlation was fixed the same way,
     * for the same reason, after a Pearson coefficient came back 0.596 instead
     * of 0.965.
     *
     * @param array<array-key, float> $a
     * @param array<array-key, float> $b
     *
     * @return array{list<float>, list<float>}
     */
    private static function assertSameShape(array $a, array $b): array
    {
        if (count($a) !== count($b)) {
            throw InvalidArgument::malformedEdge(
                'Vectors must have the same length: ' . count($a) . ' and ' . count($b)
            );
        }

        if ($a === []) {
            throw InvalidArgument::emptyDataset('a similarity');
        }

        return [array_values($a), array_values($b)];
    }
}
