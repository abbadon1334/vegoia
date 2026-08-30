<?php

declare(strict_types=1);

namespace Vegoia\Rag;

use function count;
use function max;

use Vegoia\Exception\InvalidArgument;

/**
 * Maximal Marginal Relevance re-ranking.
 *
 * Ranking purely by similarity to the query fills the context window with
 * near-duplicates: three passages that say the same thing score alike, so all
 * three get retrieved and two of them are wasted. MMR picks greedily on
 *
 *     lambda * sim(query, d) - (1 - lambda) * max sim(d, already chosen)
 *
 * so a candidate is penalised for resembling what has already been selected.
 *
 * `lambda` is the dial: 1.0 is plain relevance ranking, 0.0 ignores the query
 * and maximises spread. Around 0.5-0.7 is the usual working range. The first
 * pick is always the most relevant candidate whatever lambda is, since nothing
 * has been selected yet for it to differ from.
 *
 * @see J. Carbonell & J. Goldstein (1998), SIGIR '98, 335-336.
 */
final class MaximalMarginalRelevance
{
    /**
     * @param  list<float>                   $query
     * @param  array<array-key, list<float>> $candidates
     * @return list<array-key>               selected keys, in the order chosen
     */
    public static function select(array $query, array $candidates, int $k, float $lambda = 0.5): array
    {
        if ($k < 1) {
            throw InvalidArgument::outOfRange('Selection count', (float) $k, 1.0, INF);
        }

        if ($lambda < 0.0 || $lambda > 1.0) {
            throw InvalidArgument::outOfRange('Lambda', $lambda, 0.0, 1.0);
        }

        if ($candidates === []) {
            return [];
        }

        $relevance = [];

        foreach ($candidates as $key => $vector) {
            $relevance[$key] = Similarity::cosine($query, $vector);
        }

        // The first pick has nothing to be redundant against, so the MMR score
        // degenerates and lambda has no say. Rather than let iteration order
        // decide -- which would make lambda 0.0 open with an arbitrary
        // document -- the most relevant candidate is taken outright.
        $first = null;
        $bestRelevance = -INF;

        foreach ($relevance as $key => $score) {
            if ($score > $bestRelevance) {
                $bestRelevance = $score;
                $first = $key;
            }
        }

        if ($first === null) {
            return [];
        }

        $selected = [$first];
        $remaining = $relevance;
        unset($remaining[$first]);

        while (count($selected) < $k && $remaining !== []) {
            $bestKey = null;
            $bestScore = -INF;

            foreach ($remaining as $key => $score) {
                $redundancy = 0.0;

                foreach ($selected as $chosen) {
                    $redundancy = max(
                        $redundancy,
                        Similarity::cosine($candidates[$key], $candidates[$chosen]),
                    );
                }

                $value = $lambda * $score - (1.0 - $lambda) * $redundancy;

                // Strict comparison keeps the first candidate on a tie, and
                // iteration order here is insertion order, so the outcome does
                // not depend on hashing.
                if ($value > $bestScore) {
                    $bestScore = $value;
                    $bestKey = $key;
                }
            }

            if ($bestKey === null) {
                break;
            }

            $selected[] = $bestKey;
            unset($remaining[$bestKey]);
        }

        return $selected;
    }
}
