<?php

declare(strict_types=1);

/**
 * Drop-in replacement for a Python Leiden sidecar.
 *
 * Mirrors the signature of the `LeidenDetector` that shells out to
 * `sidecar/leiden.py`, so the swap is one class and no change at the call site:
 *
 *     detect(array $nodes, array $edges, float $resolution, int $seed): array
 *
 * with $nodes a list of string identifiers, $edges a list of
 * [from, to, weight], and the return shape
 * {communities: array<string,int>, modularity: float, count: int}.
 *
 * Run: php examples/replace_python_sidecar.php
 */

use Vegoia\Graph\Community\Leiden;
use Vegoia\Graph\Community\Quality\Modularity;
use Vegoia\Graph\Connectivity;
use Vegoia\Interop\LabelledGraph;

require __DIR__ . '/../vendor/autoload.php';

final class LeidenDetector
{
    public function __construct(
        private readonly int $minimumCommunitySize = 1,
    ) {
    }

    /**
     * @param  list<string>                       $nodes
     * @param  list<array{0: string, 1: string, 2: float}> $edges
     * @return array{communities: array<string, int>, modularity: float, count: int, orphans: list<string>}
     */
    public function detect(array $nodes, array $edges, float $resolution, int $seed): array
    {
        // Declaring the nodes keeps isolated ones in the result: a claim with
        // no links is a community of one, not an absence.
        $imported = LabelledGraph::fromEdges($edges, $nodes);
        $graph = $imported->graph;
        $index = $imported->index;

        $partition = Leiden::modularity($resolution, $seed)->partition($graph);
        $modularity = new Modularity($resolution)->of($graph, $partition);

        $orphans = [];

        if ($this->minimumCommunitySize > 1) {
            [$partition, $dropped] = $partition->withoutCommunitiesSmallerThan($this->minimumCommunitySize);

            foreach ($dropped as $node) {
                $orphans[] = $index->identifierFor($node);
            }
        }

        return [
            'communities' => $index->label($partition),
            'modularity' => $modularity,
            'count' => $partition->count(),
            'orphans' => $orphans,
        ];
    }
}

// --- A claim graph shaped like the real thing: three topics, loosely linked.
$nodes = [];
$edges = [];

foreach (['energy' => 8, 'biotech' => 6, 'policy' => 7] as $topic => $size) {
    for ($i = 0; $i < $size; $i++) {
        $nodes[] = "{$topic}-{$i}";
    }

    for ($i = 0; $i < $size; $i++) {
        for ($j = $i + 1; $j < $size; $j++) {
            $edges[] = ["{$topic}-{$i}", "{$topic}-{$j}", 0.6 + 0.4 * (($i + $j) % 3) / 3];
        }
    }
}

$nodes[] = 'unlinked-claim';
$edges[] = ['energy-0', 'policy-0', 0.15];
$edges[] = ['biotech-1', 'policy-2', 0.12];

$detector = new LeidenDetector(minimumCommunitySize: 2);

$started = hrtime(true);
$result = $detector->detect($nodes, $edges, resolution: 1.0, seed: 42);
$elapsed = (hrtime(true) - $started) / 1e6;

printf("%d claims, %d relations\n", count($nodes), count($edges));
printf(
    "%d communities, modularity %.6f, %.2f ms in-process\n\n",
    $result['count'],
    $result['modularity'],
    $elapsed
);

$grouped = [];
foreach ($result['communities'] as $identifier => $community) {
    // -1 marks a node left unassigned by the minimum-size filter. Keeping it
    // distinct from community 0 is the point of the sentinel; reporting it as
    // a community would undo that.
    if ($community >= 0) {
        $grouped[$community][] = $identifier;
    }
}
ksort($grouped);

foreach ($grouped as $community => $members) {
    sort($members);
    printf("  community %d (%d): %s\n", $community, count($members), implode(', ', $members));
}

if ($result['orphans'] !== []) {
    printf("\n  dropped below minimum size: %s\n", implode(', ', $result['orphans']));
}

// The guarantee that made Leiden worth implementing, asserted on the result.
$imported = LabelledGraph::fromEdges($edges, $nodes);
$graph = $imported->graph;
$index = $imported->index;
$allConnected = true;

foreach ($grouped as $members) {
    $internal = array_map(static fn (string $id): int => $index->nodeFor($id), $members);

    if (! Connectivity::inducesConnectedSubgraph($graph, array_values($internal))) {
        $allConnected = false;
    }
}

printf("\nevery community internally connected: %s\n", $allConnected ? 'yes' : 'NO');
printf(
    "same seed reproduces the partition:    %s\n",
    $detector->detect($nodes, $edges, 1.0, 42) === $result ? 'yes' : 'NO'
);
