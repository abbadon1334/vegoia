<?php

declare(strict_types=1);

/**
 * Timing harness for Vegoia.
 *
 * Deliberately not a framework: a benchmark that needs installing is a
 * benchmark nobody runs. Median of repeated runs rather than mean, because one
 * GC pause or one scheduler hiccup skews a mean and leaves the median alone.
 *
 * Usage: php tools/bench.php [graphdir] [--json]
 */

use Vegoia\Graph\Centrality\Betweenness;
use Vegoia\Graph\Centrality\Closeness;
use Vegoia\Graph\Centrality\PageRank;
use Vegoia\Graph\Community\Leiden;
use Vegoia\Graph\Community\Quality\Modularity;
use Vegoia\Graph\Graph;
use Vegoia\Graph\Path\Dijkstra;
use Vegoia\Stats\Descriptive;

require __DIR__ . '/../vendor/autoload.php';

/**
 * @param  callable():mixed $subject
 * @return array{median: float, min: float, runs: int, memory: int, result: mixed}
 */
function measure(callable $subject, int $runs = 7, int $warmup = 2): array
{
    for ($i = 0; $i < $warmup; $i++) {
        $subject();
    }

    $timings = [];
    $result = null;
    $before = memory_get_usage(true);
    $peak = 0;

    for ($i = 0; $i < $runs; $i++) {
        $start = hrtime(true);
        $result = $subject();
        $timings[] = (hrtime(true) - $start) / 1e6;
        $peak = max($peak, memory_get_usage(true) - $before);
    }

    sort($timings);

    return [
        'median' => $timings[intdiv(count($timings), 2)],
        'min' => $timings[0],
        'runs' => $runs,
        'memory' => $peak,
        'result' => $result,
    ];
}

/** @return array{int, list<array{int, int, float}>} */
function loadEdges(string $path): array
{
    $handle = fopen($path, 'r');

    if ($handle === false) {
        fwrite(STDERR, "Cannot read {$path}\n");

        exit(1);
    }

    $nodes = 0;
    $edges = [];

    while (($line = fgets($handle)) !== false) {
        if ($line[0] === '#') {
            $nodes = (int) explode("\t", trim($line))[1];

            continue;
        }

        [$u, $v, $w] = explode("\t", trim($line));
        $edges[] = [(int) $u, (int) $v, (float) $w];
    }

    fclose($handle);

    return [$nodes, $edges];
}

$directory = $argv[1] ?? sys_get_temp_dir() . '/vegoia-bench';
$asJson = in_array('--json', $argv, true);

$index = json_decode((string) file_get_contents("{$directory}/index.json"), true);

if (! is_array($index)) {
    fwrite(STDERR, "No benchmark graphs in {$directory}. Run tools/bench_graphs.php first.\n");

    exit(1);
}

$report = [];

if (! $asJson) {
    printf("%-8s %8s %9s %14s %12s %10s\n", 'graph', 'nodes', 'edges', 'operation', 'median ms', 'MB');
    echo str_repeat('-', 68), "\n";
}

foreach ($index as $spec) {
    [$order, $edges] = loadEdges($spec['file']);

    $build = measure(static fn (): Graph => Graph::undirected($order, $edges), runs: 3);
    /** @var Graph $graph */
    $graph = $build['result'];

    $operations = [
        'build' => $build,
        'leiden' => measure(static fn () => Leiden::modularity(seed: 42)->partition($graph), runs: 5),
        'pagerank' => measure(static fn () => (new PageRank())->of($graph), runs: 5),
        'dijkstra' => measure(static fn () => Dijkstra::distancesFrom($graph, 0), runs: 7),
    ];

    // O(nm): honest at 1k nodes, hours at 50k. Reporting a number for a size
    // nobody would run it at would be worse than reporting none.
    if ($order <= 5_000) {
        $operations['betweenness'] = measure(static fn () => Betweenness::of($graph), runs: 1, warmup: 0);
        $operations['closeness'] = measure(static fn () => Closeness::of($graph), runs: 1, warmup: 0);
    }

    $partition = Leiden::modularity(seed: 42)->partition($graph);
    $quality = (new Modularity())->of($graph, $partition);

    foreach ($operations as $name => $timing) {
        $report[] = [
            'graph' => $spec['name'],
            'nodes' => $order,
            'edges' => count($edges),
            'operation' => $name,
            'median_ms' => round($timing['median'], 3),
            'memory_mb' => round($timing['memory'] / 1048576, 1),
        ];

        if (! $asJson) {
            printf(
                "%-8s %8d %9d %14s %12.2f %10.1f\n",
                $spec['name'],
                $order,
                count($edges),
                $name,
                $timing['median'],
                $timing['memory'] / 1048576
            );
        }
    }

    if (! $asJson) {
        printf(
            "%-8s %8s %9s %14s  Q=%.6f, %d communities\n\n",
            '',
            '',
            '',
            'quality',
            $quality,
            $partition->count()
        );
    }
}

$values = [];
for ($i = 0; $i < 1_000_000; $i++) {
    $values[] = sin($i) * 1000.0 + 10_000_000.0;
}

$stats = measure(static fn () => Descriptive::of($values)->stdDev(), runs: 3);

if ($asJson) {
    echo json_encode(
        ['graph' => $report, 'stats' => ['stddev_1m_values_ms' => round($stats['median'], 3)]],
        JSON_PRETTY_PRINT
    ), "\n";
} else {
    printf("stats: standard deviation of 1,000,000 values (Welford, compensated): %.1f ms\n", $stats['median']);
}
