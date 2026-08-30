<?php

declare(strict_types=1);

/**
 * Run the benchmarks and put the numbers side by side.
 *
 * Three lanes for Vegoia's headline operations:
 *
 *   * the plain interpreter -- what `php script.php` gives you;
 *   * the tracing JIT -- one ini flag away, and 2-5x faster on these kernels;
 *   * igraph/leidenalg via python -- compiled C, the reference.
 *
 * The comparison that matters is still not "PHP versus C": it is this library
 * versus the way a PHP application actually reaches igraph today, which is a
 * subprocess -- spawn the interpreter, import igraph and leidenalg, serialise
 * the graph to JSON, parse the answer back. That fixed cost is invisible in a
 * pure-computation benchmark and is most of the wall clock on anything but a
 * very large graph.
 *
 * Usage: php tools/bench_compare.php [graphdir]
 */

$directory = $argv[1] ?? sys_get_temp_dir() . '/vegoia-bench';
$root = dirname(__DIR__);

$jitFlags = '-d opcache.enable_cli=1 -d opcache.jit=tracing -d opcache.jit_buffer_size=128M';

if (! is_file("{$directory}/index.json")) {
    echo "Generating benchmark graphs...\n";
    passthru(sprintf('php %s %s', escapeshellarg("{$root}/tools/bench_graphs.php"), escapeshellarg($directory)));
}

/** @return array{graph: list<array<string, mixed>>, stats: array<string, float>}|null */
function benchPhp(string $root, string $directory, string $flags = ''): ?array
{
    $decoded = json_decode((string) shell_exec(sprintf(
        'php %s %s %s --json',
        $flags,
        escapeshellarg("{$root}/tools/bench.php"),
        escapeshellarg($directory),
    )), true);

    return is_array($decoded) ? $decoded : null;
}

echo "Timing Vegoia (interpreter)...\n";
$plain = benchPhp($root, $directory);

echo "Timing Vegoia (tracing JIT)...\n";
$jitted = benchPhp($root, $directory, $jitFlags);

echo "Timing igraph/leidenalg...\n";
$python = json_decode((string) shell_exec(sprintf(
    'python3 %s %s --json 2>/dev/null',
    escapeshellarg("{$root}/tools/bench_reference.py"),
    escapeshellarg($directory),
)), true);

if ($plain === null || $jitted === null) {
    fwrite(STDERR, "Vegoia benchmark produced no output.\n");

    exit(1);
}

$jit = [];

foreach ($jitted['graph'] as $row) {
    $jit["{$row['graph']}/{$row['operation']}"] = $row['median_ms'];
}

$reference = [];

if (is_array($python)) {
    foreach ($python as $row) {
        $reference["{$row['graph']}/{$row['operation']}"] = $row['median_ms'];
    }
} else {
    fwrite(STDERR, "Note: igraph/leidenalg unavailable, reporting Vegoia alone.\n");
}

echo "\n";
printf(
    "%-8s %7s %8s %14s %10s %10s %10s %8s\n",
    'graph',
    'nodes',
    'edges',
    'operation',
    'Vegoia',
    '+JIT',
    'igraph',
    'JIT/C',
);
echo str_repeat('=', 82), "\n";

$previousGraph = null;

foreach ($plain['graph'] as $row) {
    if ($previousGraph !== null && $row['graph'] !== $previousGraph) {
        echo str_repeat('-', 82), "\n";
    }
    $previousGraph = $row['graph'];

    $key = "{$row['graph']}/{$row['operation']}";
    $withJit = $jit[$key] ?? null;
    $native = $reference[$key] ?? null;

    printf(
        "%-8s %7d %8d %14s %8.1fms %8s %10s %8s\n",
        $row['graph'],
        $row['nodes'],
        $row['edges'],
        $row['operation'],
        $row['median_ms'],
        $withJit === null ? '-' : sprintf('%.1fms', $withJit),
        $native === null ? '-' : sprintf('%.1fms', $native),
        $withJit === null || $native === null || $native == 0.0
            ? '-'
            : sprintf('%.2fx', $withJit / $native),
    );

    // The number an application actually pays today, against the number it
    // would pay staying in-process with the JIT on.
    $sidecar = $reference["{$row['graph']}/leiden_sidecar"] ?? null;

    if ($row['operation'] === 'leiden' && $sidecar !== null && $withJit !== null) {
        printf(
            "%-8s %7s %8s %14s %10s %8s %8.1fms %8s   <- spawn + JSON, vs +JIT\n",
            '',
            '',
            '',
            'via sidecar',
            '',
            '',
            $sidecar,
            sprintf('%.2fx', $sidecar / $withJit),
        );
    }
}

echo str_repeat('=', 82), "\n";
printf(
    "stats: stdDev over 1,000,000 values: %.1f ms interpreted, %.1f ms with JIT\n",
    $plain['stats']['stddev_1m_values_ms'],
    $jitted['stats']['stddev_1m_values_ms'],
);
echo "\nJIT/C is Vegoia with the tracing JIT against igraph's C core; the sidecar\n";
echo "row is what shelling out to Python costs relative to staying in PHP.\n";
echo "JIT flags: {$jitFlags}\n";
