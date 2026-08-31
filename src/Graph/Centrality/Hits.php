<?php

declare(strict_types=1);

namespace Vegoia\Graph\Centrality;

use function abs;
use function array_fill;
use function array_sum;
use function array_values;
use function count;

use Vegoia\Exception\InvalidArgument;
use Vegoia\Graph\Graph;

/**
 * HITS: hubs and authorities.
 *
 * Two scores instead of one, defined in terms of each other. A good authority
 * is pointed to by good hubs; a good hub points to good authorities. The
 * distinction only means something on a directed graph -- a page that
 * collects links is a different kind of important from a page that gets
 * linked -- and on an undirected graph the two scores coincide, which is
 * correct rather than a degenerate case.
 *
 * Alternating updates, which is power iteration on A'A and AA'. Both vectors
 * are normalised to sum to 1 at each step, matching networkx's `normalized`
 * default -- note that PageRank's L1 normalisation and eigenvector
 * centrality's L2 are different conventions, and mixing them up makes two
 * centralities look incomparable when they are merely scaled differently.
 *
 * @see J.M. Kleinberg (1999), "Authoritative sources in a hyperlinked
 *      environment", Journal of the ACM 46(5), 604-632.
 */
final readonly class Hits
{
    public function __construct(
        private float $tolerance = 1.0e-14,
        private int $maxIterations = 5_000,
    ) {
        if ($tolerance <= 0.0) {
            throw InvalidArgument::outOfRange('Tolerance', $tolerance, PHP_FLOAT_EPSILON, INF);
        }
    }

    /**
     * @return array{hubs: list<float>, authorities: list<float>} each summing to 1
     */
    public function of(Graph $graph): array
    {
        $order = $graph->order();

        if ($order === 0) {
            return ['hubs' => [], 'authorities' => []];
        }

        [$offsets, $targets, $weights] = $graph->csr();

        $hubs = array_fill(0, $order, 1.0 / $order);
        $authorities = array_fill(0, $order, 1.0 / $order);

        for ($iteration = 0; $iteration < $this->maxIterations; $iteration++) {
            // Authority score is what the hubs pointing at you are worth.
            $nextAuthorities = array_fill(0, $order, 0.0);

            for ($node = 0; $node < $order; $node++) {
                $value = $hubs[$node];
                $end = $offsets[$node + 1];

                for ($i = $offsets[$node]; $i < $end; $i++) {
                    $nextAuthorities[$targets[$i]] += $value * $weights[$i];
                }
            }

            // Hub score is what the authorities you point at are worth.
            $nextHubs = array_fill(0, $order, 0.0);

            for ($node = 0; $node < $order; $node++) {
                $end = $offsets[$node + 1];
                $sum = 0.0;

                for ($i = $offsets[$node]; $i < $end; $i++) {
                    $sum += $nextAuthorities[$targets[$i]] * $weights[$i];
                }

                $nextHubs[$node] = $sum;
            }

            $nextAuthorities = self::normalise(array_values($nextAuthorities));
            $nextHubs = self::normalise(array_values($nextHubs));

            // Both vectors, not one of them. The two are coupled and they do
            // not move together: on a graph where every node has in-degree 1
            // the authority vector is uniform after the first step and stays
            // put for that step, while the hub vector is still far from where
            // it is going. Measuring only the authorities declared victory
            // there after a single iteration and returned the hubs mid-flight
            // -- [0.2, 0.2, 0.4, 0.2, 0] on a graph whose answer is
            // [0, 0, 1, 0, 0].
            //
            // Every directed fixture in the suite was acyclic when this was
            // written, and on an acyclic graph the two happen to settle
            // together, so nothing caught it until a cycle was added.
            $drift = 0.0;

            for ($node = 0; $node < $order; $node++) {
                $drift += abs($nextAuthorities[$node] - $authorities[$node])
                    + abs($nextHubs[$node] - $hubs[$node]);
            }

            $authorities = $nextAuthorities;
            $hubs = $nextHubs;

            if ($drift < $this->tolerance) {
                break;
            }
        }

        /**
         * @var list<float> $hubs
         * @var list<float> $authorities
         */
        return ['hubs' => $hubs, 'authorities' => $authorities];
    }

    /**
     * @param  list<float> $vector
     * @return list<float>
     */
    private static function normalise(array $vector): array
    {
        $total = array_sum($vector);

        if ($total <= 0.0) {
            // An edgeless graph: uniform is the only defensible answer, and
            // dividing by zero is not one.
            return array_fill(0, count($vector), 1.0 / count($vector));
        }

        foreach ($vector as $index => $value) {
            $vector[$index] = $value / $total;
        }

        return $vector;
    }
}
