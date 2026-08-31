<?php

declare(strict_types=1);

/**
 * Guard against JIT-dependent results.
 *
 * The tracing JIT is the recommended way to run these kernels, so the library
 * must behave identically with and without it. That is not hypothetical: an
 * earlier optimisation of Leiden's inner loop (flat scratch buffers with a
 * version stamp in place of a per-node hash) produced a different -- and
 * worse -- partition under the JIT of PHP 8.5.10 than under the interpreter,
 * on the same seed. The tests cannot catch that class of bug from inside one
 * process, because the JIT is a process-level setting; this script catches it
 * by running the same work in both modes and comparing checksums.
 *
 * Coverage is the whole point and was once its weakness: the first version
 * checked Leiden, PageRank, descriptive statistics and least squares, and
 * missed Surprise -- which then diverged under the JIT in CI, counting 105
 * edges on a 78-edge graph. Every public entry point that does floating-point
 * work or walks the graph is checked here now, because a JIT check with a hole
 * in it is worse than none: it invites trust it has not earned.
 *
 * Exits non-zero on any divergence.
 *
 * Usage: php tools/jit_check.php
 */

$root = dirname(__DIR__);

$probe = <<<'CODE'
require $argv[1] . "/vendor/autoload.php";

use Random\Engine\Xoshiro256StarStar;
use Random\Randomizer;
use Vegoia\Graph\Centrality\Betweenness;
use Vegoia\Graph\Centrality\Closeness;
use Vegoia\Graph\Centrality\Eigenvector;
use Vegoia\Graph\Centrality\Harmonic;
use Vegoia\Graph\Centrality\Hits;
use Vegoia\Graph\Centrality\Katz;
use Vegoia\Graph\Centrality\PageRank;
use Vegoia\Graph\Clustering;
use Vegoia\Graph\Community\Agreement;
use Vegoia\Graph\Community\Leiden;
use Vegoia\Graph\Community\Quality\ConstantPotts;
use Vegoia\Graph\Community\Quality\ErdosRenyiPotts;
use Vegoia\Graph\Community\Quality\Modularity;
use Vegoia\Graph\Community\Quality\Significance;
use Vegoia\Graph\Community\Quality\Surprise;
use Vegoia\Graph\Connectivity;
use Vegoia\Graph\Graph;
use Vegoia\Graph\KCore;
use Vegoia\Graph\Partition;
use Vegoia\Graph\Path\BreadthFirst;
use Vegoia\Graph\Path\Dijkstra;
use Vegoia\Rag\MaximalMarginalRelevance;
use Vegoia\Rag\NearestNeighbours;
use Vegoia\Rag\Similarity;
use Vegoia\Stats\Correlation;
use Vegoia\Stats\Descriptive;
use Vegoia\Stats\OneWayAnova;
use Vegoia\Stats\Precision;
use Vegoia\Stats\Regression\LeastSquares;
use Vegoia\Stats\Distribution\ChiSquared;
use Vegoia\Stats\Distribution\FisherSnedecor;
use Vegoia\Stats\Distribution\Normal;
use Vegoia\Stats\Distribution\StudentT;
use Vegoia\Stats\SpecialFunction;

// A deterministic graph big enough for the JIT to warm up and trace.
$randomizer = new Randomizer(new Xoshiro256StarStar(7));
$order = 3000;
$edges = [];

for ($node = 1; $node < $order; $node++) {
    $edges[] = [$randomizer->getInt(0, $node - 1), $node, 1.0 + ($node % 7) / 7.0];
}

for ($extra = 0; $extra < 4 * $order; $extra++) {
    $edges[] = [$randomizer->getInt(0, $order - 1), $randomizer->getInt(0, $order - 1), 1.0];
}

$graph = Graph::undirected($order, $edges);
$partition = Leiden::modularity(seed: 42)->partition($graph);

$values = [];
for ($i = 0; $i < 100_000; $i++) {
    $values[] = sin($i) * 1.0e6 + 1.0e8;
}
$stats = Descriptive::of($values);

$x = [];
$y = [];
for ($i = 0; $i < 500; $i++) {
    $x[] = $i / 100.0;
    $y[] = 3.0 - 2.0 * ($i / 100.0) + (($i * 37) % 101) / 101.0;
}
$fit = LeastSquares::polynomial($x, $y, degree: 3);

$g = static fn (array $v): array => array_map(static fn (float $x): string => sprintf('%.17g', $x), $v);
$n = static fn (float $x): string => sprintf('%.17g', $x);

$directed = Graph::directed(6, [
    [0, 3, 1.0], [0, 4, 1.0], [0, 5, 1.0], [1, 3, 1.0], [1, 4, 1.0], [2, 4, 1.0], [2, 5, 1.0],
]);
$hits = (new Hits())->of($directed);
$other = Leiden::modularity(seed: 7)->partition($graph);

echo md5(json_encode([
    // community detection, all three objectives
    'membership' => $partition->membership(),
    'modularity' => $n((new Modularity())->of($graph, $partition)),
    'cpm' => $n((new ConstantPotts(0.1))->of($graph, $partition)),
    'rber' => $n((new ErdosRenyiPotts())->of($graph, $partition)),
    // the two that are scored but not optimised -- the pair that diverged
    'surprise' => $n((new Surprise())->of($graph, $partition)),
    'significance' => $n((new Significance())->of($graph, $partition)),
    'cpm_partition' => Leiden::constantPotts(0.05, 3)->partition($graph)->membership(),
    'rber_partition' => Leiden::erdosRenyiPotts(1.0, 3)->partition($graph)->membership(),
    // agreement between two runs
    'nmi' => $n(Agreement::normalisedMutualInformation($partition, $other)),
    'ari' => $n(Agreement::adjustedRandIndex($partition, $other)),
    'vi' => $n(Agreement::variationOfInformation($partition, $other)),
    // centrality
    'pagerank' => $g((new PageRank())->of($graph)),
    'pagerank_personalised' => $g((new PageRank())->of($graph, [0 => 1.0, 5 => 1.0])),
    'betweenness' => $g(Betweenness::of($graph)),
    'betweenness_weighted' => $g(Betweenness::weighted($graph)),
    'closeness' => $g(Closeness::of($graph)),
    'eigenvector' => $g((new Eigenvector())->of($graph)),
    'harmonic' => $g(Harmonic::of($graph)),
    'katz' => $g((new Katz(0.01))->of($graph)),
    'hits' => [$g($hits['hubs']), $g($hits['authorities'])],
    // structure
    'clustering' => $g(Clustering::coefficients($graph)),
    'triangles' => Clustering::triangles($graph),
    'transitivity' => $n(Clustering::transitivity($graph)),
    'core' => KCore::coreNumbers($graph),
    'components' => Connectivity::components($graph)->membership(),
    'bfs' => $g(BreadthFirst::distancesFrom($graph, 0)),
    'dijkstra' => $g(Dijkstra::distancesFrom($graph, 0)),
    // statistics, in both precisions
    'stdDev' => $n($stats->stdDev()),
    'mean' => $n($stats->mean()),
    'skewness' => $n($stats->skewness()),
    'autocorrelation' => $n($stats->autocorrelation(1)),
    'fast_stdDev' => $n($stats->with(Precision::Fast)->stdDev()),
    'fast_autocorrelation' => $n($stats->with(Precision::Fast)->autocorrelation(1)),
    'pearson' => $n(Correlation::pearson($x, $y)),
    'spearman' => $n(Correlation::spearman($x, $y)),
    'kendall' => $n(Correlation::kendall(array_slice($x, 0, 120), array_slice($y, 0, 120))),
    'anova' => $n(OneWayAnova::of([array_slice($x, 0, 100), array_slice($x, 100, 100), array_slice($y, 0, 100)])->fStatistic),
    // regression
    'coefficients' => $g($fit->coefficients),
    'standard_errors' => $g($fit->standardErrors),
    'r_squared' => $n($fit->rSquared),
    // special functions: series and continued fractions with data-dependent
    // termination, which is exactly the shape the JIT compiles aggressively
    'log_gamma' => $g(array_map(SpecialFunction::logGamma(...), [0.5, 1.5, 7.0, 100.0, 1.0e6])),
    'erf' => $g(array_map(SpecialFunction::erf(...), [-3.0, -0.5, 0.25, 1.0, 3.0])),
    'erfc' => $g(array_map(SpecialFunction::erfc(...), [0.5, 2.0, 6.0, 15.0, 26.0])),
    'gamma_p' => $g([
        SpecialFunction::regularizedGammaP(0.5, 0.25),
        SpecialFunction::regularizedGammaP(5.0, 5.0),
        SpecialFunction::regularizedGammaP(1000.0, 900.0),
    ]),
    'gamma_q' => $g([
        SpecialFunction::regularizedGammaQ(0.5, 1.0),
        SpecialFunction::regularizedGammaQ(20.0, 40.0),
        SpecialFunction::regularizedGammaQ(1000.0, 2000.0),
    ]),
    'beta' => $g([
        SpecialFunction::regularizedBeta(0.5, 0.5, 0.5),
        SpecialFunction::regularizedBeta(0.25, 2.0, 3.0),
        SpecialFunction::regularizedBeta(0.9, 100.0, 100.0),
        SpecialFunction::regularizedBeta(1.0e-6, 50.0, 5.0),
    ]),
    // distributions: both tails and the bracketed Newton behind the quantile,
    // whose iteration count depends on the data it is given
    'normal' => $g([
        new Normal()->survival(6.0), new Normal()->cumulative(-6.0),
        new Normal()->upperQuantile(1.0e-12), new Normal(3.0, 2.5)->quantile(0.05),
    ]),
    'student_t' => $g([
        new StudentT(5.0)->survival(3.0), new StudentT(1.0)->upperQuantile(1.0e-12),
        new StudentT(1000.0)->survival(1.646), new StudentT(30.0)->quantile(0.975),
    ]),
    'chi_squared' => $g([
        new ChiSquared(3.0)->survival(12.0), new ChiSquared(500.0)->survival(700.0),
        new ChiSquared(10.0)->upperQuantile(1.0e-6),
    ]),
    'fisher' => $g([
        new FisherSnedecor(3.0, 10.0)->survival(5.0),
        new FisherSnedecor(100.0, 10.0)->density(100.0),
        new FisherSnedecor(2.0, 5.0)->upperQuantile(0.01),
    ]),
    // retrieval
    'cosine' => $n(Similarity::cosine(array_slice($x, 0, 64), array_slice($y, 0, 64))),
    'knn' => array_keys(NearestNeighbours::cosine(
        array_slice($x, 0, 8),
        ['a' => array_slice($x, 0, 8), 'b' => array_slice($y, 0, 8), 'c' => array_slice($x, 8, 8)],
        2,
    )),
    'mmr' => MaximalMarginalRelevance::select(
        array_slice($x, 0, 8),
        ['a' => array_slice($x, 0, 8), 'b' => array_slice($y, 0, 8), 'c' => array_slice($x, 8, 8)],
        2,
        0.5,
    ),
])), "\n";
CODE;

$run = static function (string $flags) use ($probe, $root): string {
    $command = sprintf(
        'php %s -r %s %s',
        $flags,
        escapeshellarg($probe),
        escapeshellarg($root),
    );

    return trim((string) shell_exec($command));
};

$interpreted = $run('');
$jitted = $run('-d opcache.enable_cli=1 -d opcache.jit=tracing -d opcache.jit_buffer_size=128M');

printf("interpreter: %s\n", $interpreted);
printf("tracing JIT: %s\n", $jitted);

if ($interpreted === '' || $jitted === '') {
    fwrite(STDERR, "FAIL: a probe produced no output.\n");

    exit(1);
}

if ($interpreted !== $jitted) {
    fwrite(STDERR, "FAIL: the JIT changes results. Do not ship this.\n");

    exit(1);
}

echo "OK: identical results with and without the JIT.\n";
