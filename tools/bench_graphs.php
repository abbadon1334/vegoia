<?php

declare(strict_types=1);

/**
 * Deterministic benchmark graphs, written where both runtimes can read them.
 *
 * A planted partition (stochastic block model): nodes are split into blocks,
 * pairs inside a block are joined with probability p_in and pairs across
 * blocks with p_out. That gives community structure to actually find, unlike a
 * uniform random graph where the fastest correct answer is "one community" and
 * a benchmark measures nothing.
 *
 * Written to disk rather than generated twice so PHP and Python measure the
 * same graph, bit for bit.
 *
 * Usage: php tools/bench_graphs.php [outdir]
 */

use Random\Engine\Xoshiro256StarStar;
use Random\Randomizer;

require __DIR__ . '/../vendor/autoload.php';

$outdir = $argv[1] ?? sys_get_temp_dir() . '/vegoia-bench';

if (! is_dir($outdir) && ! mkdir($outdir, 0o777, true) && ! is_dir($outdir)) {
    fwrite(STDERR, "Cannot create {$outdir}\n");

    exit(1);
}

/** @return list<array{int, int, float}> */
function plantedPartition(int $nodes, int $blocks, float $inside, float $across, int $seed): array
{
    $randomizer = new Randomizer(new Xoshiro256StarStar($seed));
    $blockSize = intdiv($nodes, $blocks);
    $edges = [];

    // Geometric skip-sampling: instead of testing every pair, jump straight to
    // the next sampled one. The gap to the next success under probability p is
    // geometric, so the loop runs once per EDGE, not once per pair.
    $sample = static function (float $probability, int $pairs, callable $emit) use ($randomizer): void {
        if ($probability <= 0.0 || $pairs === 0) {
            return;
        }

        $logQ = log(1.0 - $probability);
        $index = -1;

        while (true) {
            $skip = (int) floor(log(1.0 - $randomizer->getFloat(0.0, 1.0)) / $logQ);
            $index += 1 + $skip;

            if ($index >= $pairs) {
                return;
            }

            $emit($index);
        }
    };

    // Unrank a pair index into (u, v) with v < u.
    $unrank = static function (int $index): array {
        $u = (int) floor((sqrt(8.0 * $index + 1.0) + 1.0) / 2.0);
        $v = $index - intdiv($u * ($u - 1), 2);

        return [$u, $v];
    };

    // Internal edges, sampled block by block. Sampling the global pair space
    // and keeping only same-block pairs would visit p_in * n^2/2 candidates --
    // a hundred million log() calls at 100k nodes -- to keep the one percent
    // that lands inside a block. Per block, the pair space is only s^2/2.
    $blockPairs = intdiv($blockSize * ($blockSize - 1), 2);

    for ($block = 0; $block < $blocks; $block++) {
        $base = $block * $blockSize;

        $sample($inside, $blockPairs, static function (int $index) use (&$edges, $base, $blockSize, $unrank): void {
            [$u, $v] = $unrank($index);

            if ($u < $blockSize && $v < $u) {
                $edges[] = [$base + $v, $base + $u, 1.0];
            }
        });
    }

    // Cross-block edges: here global sampling is fine, because p_out is tiny
    // and same-block rejects are the rare case rather than the common one.
    $pairs = intdiv($nodes * ($nodes - 1), 2);

    $sample($across, $pairs, static function (int $index) use (&$edges, $nodes, $blockSize, $unrank): void {
        [$u, $v] = $unrank($index);

        if ($u < $nodes && $v < $u && intdiv($u, $blockSize) !== intdiv($v, $blockSize)) {
            $edges[] = [$v, $u, 1.0];
        }
    });

    return $edges;
}

$specs = [
    ['name' => 'small', 'nodes' => 1_000, 'blocks' => 10, 'inside' => 0.101, 'across' => 0.0011],
    ['name' => 'medium', 'nodes' => 5_000, 'blocks' => 25, 'inside' => 0.050, 'across' => 0.00021],
    ['name' => 'large', 'nodes' => 20_000, 'blocks' => 50, 'inside' => 0.025, 'across' => 0.000051],
    ['name' => 'huge', 'nodes' => 50_000, 'blocks' => 100, 'inside' => 0.020, 'across' => 0.00002],
    ['name' => '100k', 'nodes' => 100_000, 'blocks' => 200, 'inside' => 0.020, 'across' => 0.00001],
];

$index = [];

foreach ($specs as $spec) {
    $edges = plantedPartition(
        $spec['nodes'],
        $spec['blocks'],
        $spec['inside'],
        $spec['across'],
        seed: 20260830,
    );

    $path = "{$outdir}/{$spec['name']}.tsv";
    $handle = fopen($path, 'w');

    if ($handle === false) {
        fwrite(STDERR, "Cannot write {$path}\n");

        exit(1);
    }

    fwrite($handle, "# nodes\t{$spec['nodes']}\n");

    foreach ($edges as [$u, $v, $w]) {
        fwrite($handle, "{$u}\t{$v}\t{$w}\n");
    }

    fclose($handle);

    $index[] = ['name' => $spec['name'], 'nodes' => $spec['nodes'], 'edges' => count($edges), 'file' => $path];
    printf(
        "%-8s n=%6d m=%8d  blocks=%3d  -> %s\n",
        $spec['name'],
        $spec['nodes'],
        count($edges),
        $spec['blocks'],
        $path
    );
}

file_put_contents("{$outdir}/index.json", json_encode($index, JSON_PRETTY_PRINT) . "\n");
echo "\nindex -> {$outdir}/index.json\n";
