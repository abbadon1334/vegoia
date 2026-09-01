<?php

declare(strict_types=1);

namespace Vegoia\Graph\Centrality;

use function abs;
use function array_fill;
use function max;
use function sqrt;

use Vegoia\Exception\DidNotConverge;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Graph\Graph;

/**
 * Katz centrality: influence that decays with distance.
 *
 *     x = alpha * A x + beta
 *
 * Every walk into a node counts, discounted by alpha^length, plus a baseline
 * beta everyone gets for existing. It sits between degree, which counts only
 * walks of length one, and eigenvector centrality, which counts all walks
 * without decay.
 *
 * The baseline is not decoration: it is what makes this usable where
 * eigenvector centrality is not. On a disconnected graph the principal
 * eigenvector collapses onto one component and everything else scores zero;
 * beta gives every node a floor, so peripheral components keep meaningful
 * relative scores.
 *
 * alpha must stay below 1 / lambda_max or the series diverges -- longer walks
 * would count for more than shorter ones, which is not a centrality. The
 * default 0.05 is safe on most graphs; `isConvergent()` checks a given graph
 * rather than leaving it to chance.
 *
 * @see L. Katz (1953), "A new status index derived from sociometric analysis",
 *      Psychometrika 18(1), 39-43.
 */
final readonly class Katz
{
    public function __construct(
        private float $alpha = 0.05,
        private float $beta = 1.0,
        private float $tolerance = 1.0e-12,
        private int $maxIterations = 10_000,
    ) {
        if ($alpha <= 0.0) {
            throw InvalidArgument::outOfRange('Alpha', $alpha, PHP_FLOAT_EPSILON, INF);
        }
    }

    /** Whether alpha is small enough for the series to converge on this graph. */
    public function isConvergent(Graph $graph): bool
    {
        return $this->alpha < self::criticalAlpha($graph);
    }

    /**
     * The spectral radius: the largest eigenvalue of the adjacency matrix in
     * magnitude, which is what bounds alpha.
     *
     * Computed by power iteration rather than bounded by the maximum strength.
     * That bound (Gershgorin) is a guarantee but a loose one -- on Zachary it
     * claims 48 where the true value is near 21 -- and using it would reject
     * alphas that converge perfectly well, which is worse than useless in a
     * validity check. The shift is the same device Eigenvector needs, and for
     * the same reason: without it a bipartite graph oscillates forever.
     */
    public static function spectralRadius(Graph $graph): float
    {
        $order = $graph->order();

        if ($order === 0) {
            return 0.0;
        }

        [$offsets, $targets, $weights] = $graph->csr();

        $shift = 0.0;
        for ($node = 0; $node < $order; $node++) {
            $shift = max($shift, $graph->strength($node));
        }

        if ($shift === 0.0) {
            return 0.0;
        }

        $vector = array_fill(0, $order, 1.0 / sqrt((float) $order));
        $eigenvalue = 0.0;
        $converged = false;
        $movement = INF;

        // A thousand steps of a shifted power iteration. Shifting is what makes
        // that enough: without it the convergence rate is the ratio of the two
        // largest eigenvalues in absolute value, which is 1 on a bipartite
        // graph and never converges at all.
        for ($iteration = 0; $iteration < 1000; $iteration++) {
            $next = array_fill(0, $order, 0.0);

            for ($node = 0; $node < $order; $node++) {
                $value = $vector[$node];
                $next[$node] += $shift * $value;
                $end = $offsets[$node + 1];

                for ($i = $offsets[$node]; $i < $end; $i++) {
                    $next[$targets[$i]] += $value * $weights[$i];
                }
            }

            $norm = 0.0;
            foreach ($next as $value) {
                $norm += $value * $value;
            }
            $norm = sqrt($norm);

            if ($norm === 0.0) {
                return 0.0;
            }

            for ($node = 0; $node < $order; $node++) {
                $next[$node] /= $norm;
            }

            // The shifted eigenvalue is the norm; undo the shift.
            $previous = $eigenvalue;
            $eigenvalue = $norm - $shift;
            $vector = $next;

            $movement = abs($eigenvalue - $previous);

            if ($movement < 1.0e-12 * max(1.0, abs($eigenvalue))) {
                $converged = true;

                break;
            }
        }

        // Refusing rather than returning the estimate it happened to reach.
        // This number decides whether the caller's alpha is inside the radius
        // of convergence, so an estimate that has not settled can wave through
        // an alpha whose series diverges -- and the scores that come back from
        // that are not wrong by a little.
        if (! $converged) {
            throw DidNotConverge::after(
                "Katz's estimate of the largest eigenvalue",
                1000,
                $movement,
                1.0e-12 * max(1.0, abs($eigenvalue)),
            );
        }

        return abs($eigenvalue);
    }

    /**
     * The largest alpha that still converges on this graph, 1 / lambda_max.
     *
     * Half of it is a reasonable working value: safely inside the bound, and
     * not so small that every score collapses onto the beta baseline.
     */
    public static function criticalAlpha(Graph $graph): float
    {
        $radius = self::spectralRadius($graph);

        return $radius === 0.0 ? INF : 1.0 / $radius;
    }

    /** @return list<float> L2-normalised, matching networkx's default */
    public function of(Graph $graph): array
    {
        $order = $graph->order();

        if ($order === 0) {
            return [];
        }

        // Refused rather than returned as NaN. A diverging series produces
        // inf, then inf/inf, and the caller receives a vector of NaN with no
        // indication of why -- on a weighted graph, where lambda_max is large,
        // this is the common case rather than an edge case.
        if (! $this->isConvergent($graph)) {
            throw InvalidArgument::outOfRange(
                'Katz alpha must stay below 1 / lambda_max, which for this graph is '
                . self::criticalAlpha($graph) . '; the given alpha',
                $this->alpha,
                0.0,
                self::criticalAlpha($graph),
            );
        }

        [$offsets, $targets, $weights] = $graph->csr();

        $score = array_fill(0, $order, 0.0);

        $converged = false;
        $drift = INF;

        for ($iteration = 0; $iteration < $this->maxIterations; $iteration++) {
            $next = array_fill(0, $order, $this->beta);

            for ($node = 0; $node < $order; $node++) {
                $value = $score[$node];

                if ($value === 0.0) {
                    continue;
                }

                $end = $offsets[$node + 1];
                $scaled = $this->alpha * $value;

                for ($i = $offsets[$node]; $i < $end; $i++) {
                    $next[$targets[$i]] += $scaled * $weights[$i];
                }
            }

            $drift = 0.0;

            for ($node = 0; $node < $order; $node++) {
                $drift += abs($next[$node] - $score[$node]);
            }

            $score = $next;

            if ($drift < $order * $this->tolerance) {
                $converged = true;

                break;
            }
        }

        if (! $converged) {
            throw DidNotConverge::after('Katz centrality', $this->maxIterations, $drift, $order * $this->tolerance);
        }

        $norm = 0.0;
        foreach ($score as $value) {
            $norm += $value * $value;
        }
        $norm = sqrt($norm);

        if ($norm > 0.0) {
            for ($node = 0; $node < $order; $node++) {
                $score[$node] /= $norm;
            }
        }

        /** @var list<float> $score */
        return $score;
    }
}
