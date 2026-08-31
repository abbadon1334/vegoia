<?php

declare(strict_types=1);

namespace Vegoia\Graph;

use function array_slice;
use function arsort;
use function count;
use function log;

use Vegoia\Exception\InvalidArgument;

/**
 * Which edges are probably missing.
 *
 * The question a retrieval-augmented system asks constantly without naming it:
 * given the entity just retrieved, which others is it likely related to even
 * though nobody wrote the edge down? A knowledge graph built by extraction is
 * always incomplete, and the gaps are not random -- they are the relations
 * nobody bothered to state because they were obvious.
 *
 * Every measure here works from shared neighbourhoods, except the one that
 * deliberately does not; see LinkMeasure for what each believes.
 *
 * Two entry points, and the difference between them is not cosmetic. score()
 * answers about one pair. rank() answers "which nodes should this one be
 * joined to", and reaches its candidates by walking two hops rather than by
 * scoring every pair -- which is the difference between the neighbourhood of a
 * node and the whole graph, and on anything larger than a toy it is the
 * difference between usable and not.
 */
final class LinkPrediction
{
    /** Score one pair. Defined whether or not the two are already joined. */
    public static function score(Graph $graph, int $from, int $to, LinkMeasure $measure): float
    {
        self::assertUndirected($graph);

        [$offsets, $targets] = $graph->csr();

        if ($measure === LinkMeasure::PreferentialAttachment) {
            return (float) ($graph->degree($from) * $graph->degree($to));
        }

        // Both neighbourhoods are sorted runs of the same array, so the shared
        // part is a merge rather than a lookup per element.
        $shared = [];
        $i = $offsets[self::assertNode($graph, $from)];
        $j = $offsets[self::assertNode($graph, $to)];
        $endI = $offsets[$from + 1];
        $endJ = $offsets[$to + 1];
        $union = 0;

        while ($i < $endI && $j < $endJ) {
            $a = $targets[$i];
            $b = $targets[$j];

            if ($a === $b) {
                $shared[] = $a;
                $i++;
                $j++;
            } elseif ($a < $b) {
                $i++;
            } else {
                $j++;
            }

            $union++;
        }

        $union += ($endI - $i) + ($endJ - $j);

        // ResourceAllocation is the default arm rather than a named one only
        // because preferential attachment returned above, which leaves it as
        // the single remaining case; naming it as well would be a branch that
        // cannot be taken.
        return match ($measure) {
            LinkMeasure::CommonNeighbours => (float) count($shared),
            LinkMeasure::Jaccard => $union === 0 ? 0.0 : count($shared) / $union,
            LinkMeasure::AdamicAdar => self::weighted($graph, $shared, logarithmic: true),
            default => self::weighted($graph, $shared, logarithmic: false),
        };
    }

    /**
     * The nodes this one should probably be joined to, best first.
     *
     * Already-joined neighbours are left out: the question is what is missing,
     * not what is there. So is the node itself.
     *
     * Candidates come from two hops away -- the neighbours of my neighbours --
     * because every measure but one scores zero without a shared neighbour,
     * and a zero is not a prediction. Preferential attachment is the
     * exception and has to consider everybody, which is stated in the
     * enum rather than assumed here.
     *
     * @param int|null $limit how many to return; null for all of them
     *
     * @return array<int, float> node => score, in descending order
     */
    public static function rank(
        Graph $graph,
        int $node,
        LinkMeasure $measure,
        ?int $limit = null,
    ): array {
        self::assertUndirected($graph);
        self::assertNode($graph, $node);

        if ($limit !== null && $limit < 1) {
            throw InvalidArgument::outOfRange('Result limit', (float) $limit, 1.0, INF);
        }

        $scores = [];

        foreach (self::candidates($graph, $node, $measure) as $candidate) {
            $score = self::score($graph, $node, $candidate, $measure);

            if ($score > 0.0) {
                $scores[$candidate] = $score;
            }
        }

        // Descending by score; ties keep the order the candidates were found
        // in, which is ascending by node, so the result is deterministic.
        arsort($scores, SORT_NUMERIC);

        return $limit === null ? $scores : array_slice($scores, 0, $limit, preserve_keys: true);
    }

    /**
     * The nodes worth scoring at all.
     *
     * @return list<int>
     */
    private static function candidates(Graph $graph, int $node, LinkMeasure $measure): array
    {
        [$offsets, $targets] = $graph->csr();

        if ($measure->scoresStrangers()) {
            $all = [];

            for ($other = 0; $other < $graph->order(); $other++) {
                if ($other !== $node && ! $graph->hasEdge($node, $other)) {
                    $all[] = $other;
                }
            }

            return $all;
        }

        $seen = [];

        for ($i = $offsets[$node]; $i < $offsets[$node + 1]; $i++) {
            $neighbour = $targets[$i];

            for ($j = $offsets[$neighbour]; $j < $offsets[$neighbour + 1]; $j++) {
                $candidate = $targets[$j];

                if ($candidate !== $node && ! isset($seen[$candidate])) {
                    $seen[$candidate] = true;
                }
            }
        }

        $candidates = [];

        foreach ($seen as $candidate => $_) {
            if (! $graph->hasEdge($node, $candidate)) {
                $candidates[] = $candidate;
            }
        }

        sort($candidates);

        return $candidates;
    }

    /**
     * Sum over the shared neighbours of 1/log(degree) or 1/degree.
     *
     * A neighbour of degree 1 is impossible here -- it would have to be shared
     * by two nodes and so have degree at least 2 -- but a self-loop can make
     * log(degree) zero, and dividing by it would return infinity for a
     * structure that is not evidence of anything.
     *
     * @param list<int> $shared
     */
    private static function weighted(Graph $graph, array $shared, bool $logarithmic): float
    {
        $total = 0.0;

        foreach ($shared as $neighbour) {
            $degree = (float) $graph->degree($neighbour);
            $weight = $logarithmic ? log($degree) : $degree;

            if ($weight > 0.0) {
                $total += 1.0 / $weight;
            }
        }

        return $total;
    }

    private static function assertNode(Graph $graph, int $node): int
    {
        if (! $graph->hasNode($node)) {
            throw InvalidArgument::nodeOutOfRange($node, $graph->order());
        }

        return $node;
    }

    private static function assertUndirected(Graph $graph): void
    {
        if ($graph->isDirected()) {
            throw InvalidArgument::directedNotSupported(
                'Link prediction',
                'the directed measures count in-neighbourhoods and out-neighbourhoods separately, '
                . 'which is a different definition rather than the same one applied twice.',
            );
        }
    }
}
