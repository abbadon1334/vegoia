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

// Extensions that override zend_execute_ex switch the JIT off, and PHP says so
// on stderr and then runs anyway. pcov is one of them, and it is installed on
// any machine that has run the coverage job -- so this benchmark was timing
// the interpreter in both lanes and reporting a speedup of 1.0x, exactly the
// failure jit_check.php was fixed for. Both lanes disable it: leaving a
// profiler hooked into the interpreter lane would penalise it for something
// nobody runs in production, and leaving it in the JIT lane means there is no
// JIT lane.
$disable = '-d pcov.enabled=0 -d xdebug.mode=off';
$jitFlags = $disable . ' -d opcache.enable_cli=1 -d opcache.jit=tracing -d opcache.jit_buffer_size=128M';

// Asked, not assumed. The flags above are a request; opcache is the authority
// on whether it was granted.
$jitState = trim((string) shell_exec(sprintf(
    'php %s -r %s 2>/dev/null',
    $jitFlags,
    escapeshellarg(
        '$s = function_exists("opcache_get_status") ? @opcache_get_status(false) : null;'
        . 'echo ($s["jit"]["on"] ?? false) ? "on" : "off";'
    ),
)));

if ($jitState !== 'on') {
    fwrite(STDERR, "The JIT would not turn on, so the +JIT column would repeat the first one.\n");
    fwrite(STDERR, "Usually an extension that overrides zend_execute_ex, or opcache missing.\n");

    exit(1);
}

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
$plain = benchPhp($root, $directory, $disable);

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
$referenceStats = [];

if (is_array($python) && isset($python['graph'])) {
    foreach ($python['graph'] as $row) {
        $reference["{$row['graph']}/{$row['operation']}"] = $row['median_ms'];
    }

    $referenceStats = $python['stats'] ?? [];
} else {
    fwrite(STDERR, "Note: the reference libraries are unavailable, reporting Vegoia alone.\n");
}

echo "\n";
printf(
    "%-8s %7s %8s %14s %10s %10s %10s %10s %8s\n",
    'graph',
    'nodes',
    'edges',
    'operation',
    'Vegoia',
    '+JIT',
    'igraph',
    'networkx',
    'JIT/C',
);
echo str_repeat('=', 93), "\n";

$previousGraph = null;

foreach ($plain['graph'] as $row) {
    if ($previousGraph !== null && $row['graph'] !== $previousGraph) {
        echo str_repeat('-', 93), "\n";
    }
    $previousGraph = $row['graph'];

    $key = "{$row['graph']}/{$row['operation']}";
    $withJit = $jit[$key] ?? null;
    $native = $reference[$key] ?? null;

    // networkx names the same operations with an nx_ prefix, and calls
    // community detection Louvain because it has no Leiden.
    $pure = $reference["{$row['graph']}/nx_" . ($row['operation'] === 'leiden' ? 'louvain' : $row['operation'])] ?? null;

    printf(
        "%-8s %7d %8d %14s %8.1fms %8s %10s %10s %8s\n",
        $row['graph'],
        $row['nodes'],
        $row['edges'],
        $row['operation'],
        $row['median_ms'],
        $withJit === null ? '-' : sprintf('%.1fms', $withJit),
        $native === null ? '-' : sprintf('%.1fms', $native),
        $pure === null ? '-' : sprintf('%.1fms', $pure),
        $withJit === null || $native === null || $native == 0.0
            ? '-'
            : sprintf('%.2fx', $withJit / $native),
    );

    // The number an application actually pays today, against the number it
    // would pay staying in-process with the JIT on.
    $sidecar = $reference["{$row['graph']}/leiden_sidecar"] ?? null;

    if ($row['operation'] === 'leiden' && $sidecar !== null && $withJit !== null) {
        printf(
            "%-8s %7s %8s %14s %10s %8s %8.1fms %10s %8s   <- spawn + JSON, vs +JIT\n",
            '',
            '',
            '',
            'via sidecar',
            '',
            '',
            $sidecar,
            '',
            sprintf('%.2fx', $sidecar / $withJit),
        );
    }
}

echo str_repeat('=', 93), "\n";

echo "\nJIT/C is Vegoia with the tracing JIT against igraph's C core; networkx is\n";
echo "the same operation in a library written in the host language, which is the\n";
echo "position Vegoia is in, and its community detection is Louvain because there\n";
echo "is no Leiden in it. The sidecar row is what shelling out to Python costs\n";
echo "relative to staying in PHP.\n";

printf(
    "\n%-24s %10s %10s %12s %8s\n",
    'statistics',
    'Vegoia',
    '+JIT',
    'numpy/scipy',
    'JIT/C',
);
echo str_repeat('=', 68), "\n";

foreach ($plain['stats'] as $name => $ms) {
    $withJit = $jitted['stats'][$name] ?? null;
    $native = $referenceStats[$name] ?? null;

    printf(
        "%-24s %8.1fms %8s %10s %8s\n",
        $name,
        $ms,
        $withJit === null ? '-' : sprintf('%.1fms', $withJit),
        $native === null ? '-' : sprintf('%.2fms', $native),
        $withJit === null || $native === null || $native == 0.0
            ? '-'
            : sprintf('%.0fx', $withJit / $native),
    );
}

echo str_repeat('=', 68), "\n";
echo "\nnumpy and SciPy work on a whole array inside compiled code while Vegoia\n";
echo "walks it in PHP, so these ratios are large and are not meant to be read as\n";
echo "a defect. They are the comparison a caller faces. Note which np.std is\n";
echo "being beaten by: it is a plain two-pass sum, the same arithmetic as\n";
echo "Precision::Fast, not the compensated default -- see the accuracy report\n";
echo "for what the default buys.\n";
echo "JIT flags: {$jitFlags}\n";
