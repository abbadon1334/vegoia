<?php

declare(strict_types=1);

namespace Vegoia\Graph\Centrality;

use function abs;
use function array_fill;
use function max;
use function sqrt;

use Vegoia\Exception\InvalidArgument;
use Vegoia\Graph\Graph;

/**
 * Eigenvector centrality: importance defined recursively.
 *
 * A node is important in proportion to how important its neighbours are, which
 * as an equation is x = (1/lambda) A x -- the principal eigenvector of the
 * adjacency matrix. Where degree counts connections, this weighs them: one
 * edge to a hub can outrank ten edges to nobodies.
 *
 * Found by power iteration, which is the eigenvector's own definition used as
 * an algorithm: repeatedly multiply by A and renormalise, and everything
 * except the principal direction dies off at a rate set by the ratio of the
 * two largest eigenvalues.
 *
 * Plain power iteration fails on bipartite graphs, and fails by oscillating
 * rather than by diverging, so it looks like slow convergence. A bipartite
 * adjacency matrix has a symmetric spectrum -- for every eigenvalue lambda
 * there is a -lambda -- so the two largest are equal in magnitude, nothing
 * decays, and the vector flips between two states forever. A star graph is
 * bipartite; so is any two-mode network. Iterating on (A + cI) instead fixes
 * it: adding c to every eigenvalue breaks the symmetry while leaving the
 * eigenvectors untouched, since (A + cI)x = (lambda + c)x for the same x.
 *
 * Known limitation. On a disconnected graph the principal eigenvector
 * concentrates entirely on the densest component and every other node scores
 * essentially zero -- correct, and rarely what the caller wanted. This is why
 * PageRank exists, and why it is the better default: teleportation makes the
 * problem irreducible and every component gets a share.
 *
 * @see P. Bonacich (1987), "Power and Centrality: A Family of Measures",
 *      American Journal of Sociology 92(5), 1170-1182.
 */
final readonly class Eigenvector
{
    public function __construct(
        private float $tolerance = 1.0e-12,
        private int $maxIterations = 10_000,
    ) {
        if ($tolerance <= 0.0) {
            throw InvalidArgument::outOfRange('Tolerance', $tolerance, PHP_FLOAT_EPSILON, INF);
        }
    }

    /** @return list<float> L2-normalised, non-negative */
    public function of(Graph $graph): array
    {
        $order = $graph->order();

        if ($order === 0) {
            return [];
        }

        [$offsets, $targets, $weights] = $graph->csr();

        // The spectral shift. Any value exceeding |lambda_min| works, and the
        // largest strength bounds every eigenvalue by Gershgorin, so it is
        // certainly large enough. Bigger shifts converge more slowly -- the
        // eigenvalue ratios crowd towards 1 -- so this takes the smallest
        // bound that is guaranteed rather than a comfortable overestimate.
        $shift = 0.0;

        for ($node = 0; $node < $order; $node++) {
            $shift = max($shift, $graph->strength($node));
        }

        if ($shift === 0.0) {
            // No edges: every node is equally unimportant, and the zero
            // vector is not a centrality.
            return array_fill(0, $order, 1.0 / sqrt((float) $order));
        }

        // Starting uniform and positive keeps the iteration inside the cone
        // where Perron-Frobenius applies, so the limit is the non-negative
        // principal eigenvector rather than an arbitrary one.
        $score = array_fill(0, $order, 1.0 / sqrt((float) $order));

        for ($iteration = 0; $iteration < $this->maxIterations; $iteration++) {
            $next = array_fill(0, $order, 0.0);

            for ($node = 0; $node < $order; $node++) {
                $value = $score[$node];
                $end = $offsets[$node + 1];

                // The cI part of (A + cI).
                $next[$node] += $shift * $value;

                for ($i = $offsets[$node]; $i < $end; $i++) {
                    $next[$targets[$i]] += $value * $weights[$i];
                }
            }

            $norm = 0.0;
            foreach ($next as $value) {
                $norm += $value * $value;
            }
            $norm = sqrt($norm);

            $drift = 0.0;

            for ($node = 0; $node < $order; $node++) {
                $next[$node] /= $norm;
                $drift += abs($next[$node] - $score[$node]);
            }

            $score = $next;

            if ($drift < $order * $this->tolerance) {
                break;
            }
        }

        /** @var list<float> $score */
        return $score;
    }
}
