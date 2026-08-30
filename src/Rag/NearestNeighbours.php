<?php

declare(strict_types=1);

namespace Vegoia\Rag;

use function array_slice;
use function uksort;

use Vegoia\Exception\InvalidArgument;

/**
 * Exact k-nearest-neighbour search by cosine similarity.
 *
 * Exact, not approximate: it compares against every candidate. For the
 * thousands-to-low-millions of vectors a PHP process realistically holds in
 * memory that is the right trade -- no index to build, no recall to tune, no
 * drift between the index and the data. Past that scale the answer is a vector
 * database, and this class is the thing you check its recall against.
 *
 * Ties break on the candidate key, so a run is reproducible. Without that, two
 * identical documents swap places between requests according to hash order,
 * and a "stable" ranking silently is not one.
 */
final class NearestNeighbours
{
    /**
     * @param  list<float>                    $query
     * @param  array<array-key, list<float>>  $candidates
     * @return array<array-key, float>        key => similarity, best first
     */
    public static function cosine(array $query, array $candidates, int $k): array
    {
        if ($k < 1) {
            throw InvalidArgument::outOfRange('Neighbour count', (float) $k, 1.0, INF);
        }

        if ($candidates === []) {
            return [];
        }

        $scores = [];

        foreach ($candidates as $key => $vector) {
            $scores[$key] = Similarity::cosine($query, $vector);
        }

        uksort($scores, static function (int|string $left, int|string $right) use ($scores): int {
            return $scores[$right] <=> $scores[$left] ?: $left <=> $right;
        });

        return array_slice($scores, 0, $k, preserve_keys: true);
    }
}
