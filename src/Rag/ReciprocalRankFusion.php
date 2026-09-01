<?php

declare(strict_types=1);

namespace Vegoia\Rag;

use function array_key_exists;
use function rsort;

use Vegoia\Exception\InvalidArgument;

/**
 * One ranking out of several, without a shared scale.
 *
 * The problem it solves is that a BM25 score and a cosine similarity are not
 * comparable and do not become comparable by rescaling: normalising each to
 * [0, 1] makes the top hit of a search that found nothing worth having look
 * exactly like the top hit of one that did. Reciprocal Rank Fusion throws the
 * scores away and keeps only the positions, so a document rises by being near
 * the top of several lists rather than by scoring highly on whichever scale
 * happened to be generous.
 *
 *     score(d) = sum over rankings of 1 / (k + rank(d))
 *
 * k damps the top of each list. At 60 the first and second places differ by
 * 1/61 - 1/62, about a quarter of a percent, so no single engine can carry a
 * document on its own -- which is the point. Raising k towards infinity
 * flattens the method into a plain vote; lowering it towards zero hands the
 * answer to whichever list is most confident.
 *
 * @see G.V. Cormack, C.L.A. Clarke & S. Buettcher (2009), "Reciprocal Rank
 *      Fusion outperforms Condorcet and individual Rank Learning Methods",
 *      SIGIR '09, 758-759.
 */
final class ReciprocalRankFusion
{
    /**
     * The constant from Cormack, Clarke and Buettcher, chosen by them on TREC
     * data and not re-derived here.
     */
    public const int DEFAULT_K = 60;

    /**
     * Fuse several rankings into one.
     *
     * There is deliberately no "how many to return" argument. Cormack's
     * constant is universally written k, and NearestNeighbours::cosine()
     * already uses $k for the number of results; giving this class both
     * meanings would be a collision nobody reads twice. Take a prefix with
     * array_slice($fused, 0, 10, preserve_keys: true).
     *
     * @param list<list<array-key>> $rankings each best first
     *
     * @return array<array-key, float> key => fused score, best first
     */
    public static function fuse(array $rankings, int $k = self::DEFAULT_K): array
    {
        if ($k < 0) {
            throw InvalidArgument::outOfRange('Rank fusion constant', (float) $k, 0.0, INF);
        }

        /** @var array<array-key, list<int>> $positions */
        $positions = [];

        foreach ($rankings as $ranking) {
            $seen = [];

            foreach ($ranking as $index => $key) {
                // The first occurrence wins. A document listed twice by one
                // retriever would otherwise collect double credit from a
                // single opinion.
                if (array_key_exists($key, $seen)) {
                    continue;
                }

                $seen[$key] = true;

                // Ranks are 1-based. The paper writes 1/(k + r) with r from 1
                // and chose k = 60 against that; reading it 0-based gives the
                // top document 1/60 instead of 1/61, which is a different
                // scoring function wearing the same constant.
                $positions[$key][] = $index + 1;
            }
        }

        $scores = [];

        foreach ($positions as $key => $ranks) {
            // Summed in a canonical order -- largest rank first, so smallest
            // contribution first -- and that is not a micro-optimisation.
            // Floating addition is not associative, so two documents appearing
            // at the same set of positions can otherwise differ by one unit in
            // the last place purely because the rankings listed them in
            // different orders. The tie-break below would then never fire and
            // the ordering would be decided by a rounding difference.
            rsort($ranks);

            $total = 0.0;

            foreach ($ranks as $rank) {
                $total += 1.0 / ($k + $rank);
            }

            $scores[$key] = $total;
        }

        // Descending by score, ties broken on the key ascending -- the same
        // convention as NearestNeighbours, deliberately. Two ranking APIs in
        // one namespace that break ties differently is a bug generator.
        uksort($scores, static function ($a, $b) use ($scores): int {
            return $scores[$b] <=> $scores[$a] ?: $a <=> $b;
        });

        return $scores;
    }
}
