<?php

declare(strict_types=1);

namespace Vegoia\Rag;

use function array_flip;
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
     * @param list<float> $a
     * @param list<float> $b
     */
    public static function cosine(array $a, array $b): float
    {
        self::assertSameShape($a, $b);

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
     * @param list<float> $a
     * @param list<float> $b
     */
    public static function dot(array $a, array $b): float
    {
        self::assertSameShape($a, $b);

        $sum = new CompensatedSum();

        foreach ($a as $i => $value) {
            $sum->add($value * $b[$i]);
        }

        return $sum->value();
    }

    /**
     * @param list<float> $a
     * @param list<float> $b
     */
    public static function euclidean(array $a, array $b): float
    {
        self::assertSameShape($a, $b);

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
     * @param list<float> $a
     * @param list<float> $b
     */
    private static function assertSameShape(array $a, array $b): void
    {
        if (count($a) !== count($b)) {
            throw InvalidArgument::malformedEdge(
                'Vectors must have the same length: ' . count($a) . ' and ' . count($b)
            );
        }

        if ($a === []) {
            throw InvalidArgument::emptyDataset('a similarity');
        }
    }
}
