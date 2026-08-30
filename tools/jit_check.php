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
 * Exits non-zero on any divergence.
 *
 * Usage: php tools/jit_check.php
 */

$root = dirname(__DIR__);

$probe = <<<'CODE'
require $argv[1] . "/vendor/autoload.php";

use Random\Engine\Xoshiro256StarStar;
use Random\Randomizer;
use Vegoia\Graph\Centrality\PageRank;
use Vegoia\Graph\Community\Leiden;
use Vegoia\Graph\Community\Quality\Modularity;
use Vegoia\Graph\Graph;
use Vegoia\Stats\Descriptive;
use Vegoia\Stats\Regression\LeastSquares;

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

echo md5(json_encode([
    'membership' => $partition->membership(),
    'modularity' => sprintf('%.17g', (new Modularity())->of($graph, $partition)),
    'pagerank' => array_map(static fn (float $v): string => sprintf('%.17g', $v), (new PageRank())->of($graph)),
    'stdDev' => sprintf('%.17g', $stats->stdDev()),
    'coefficients' => array_map(static fn (float $v): string => sprintf('%.17g', $v), $fit->coefficients),
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
