<?php

declare(strict_types=1);

namespace Vegoia\Tests\Support;

use function array_map;
use function file_get_contents;
use function is_array;
use function json_decode;

use const JSON_THROW_ON_ERROR;

use RuntimeException;
use Vegoia\Graph\Graph;
use Vegoia\Graph\Partition;

/**
 * A golden fixture produced by `tools/generate_graph_fixtures.py`.
 *
 * The expected values come from the reference implementations -- leidenalg
 * (by the author of the Leiden algorithm) and networkx -- never from Vegoia
 * itself. Regenerating them requires those Python packages; the JSON is
 * committed so the PHP suite runs without them.
 */
final readonly class GraphFixture
{
    /**
     * @param list<array{int, int, float}> $edges
     * @param array<string, mixed>         $expected
     */
    private function __construct(
        public string $name,
        public string $note,
        public int $nodes,
        public array $edges,
        public array $expected,
    ) {
    }

    public static function load(string $name): self
    {
        return self::loadFrom("graph/{$name}.json");
    }

    private static function loadFrom(string $relative): self
    {
        $path = Paths::fixture($relative);
        $raw = file_get_contents($path);

        if ($raw === false) {
            throw new RuntimeException(
                "Missing graph fixture {$path}. Regenerate it with the matching script in tools/."
            );
        }

        $data = json_decode($raw, associative: true, flags: JSON_THROW_ON_ERROR);

        if (! is_array($data)) {
            throw new RuntimeException("Malformed graph fixture {$path}");
        }

        /** @var array{name: string, note: string, nodes: int, edges: list<array{int, int, float}>, expected?: array<string, mixed>} $data */
        // Labelled fixtures put their expectations at the top level rather
        // than under 'expected', since the ground truth is the fixture's
        // point rather than one property of it.
        $expected = $data['expected'] ?? $data;

        return new self($data['name'], $data['note'], $data['nodes'], $data['edges'], $expected);
    }

    /** A fixture that carries the answer, not just a reference implementation's opinion of it. */
    public static function labelled(string $name): self
    {
        return self::loadFrom("labelled/{$name}.json");
    }

    /** @return list<string> */
    public static function labelledNames(): array
    {
        return self::namesFrom('labelled/index.json');
    }

    public static function directed(string $name): self
    {
        return self::loadFrom("directed/{$name}.json");
    }

    /** @return list<string> */
    public static function directedNames(): array
    {
        return self::namesFrom('directed/index.json');
    }

    public function directedGraph(): Graph
    {
        return Graph::directed($this->nodes, $this->edges);
    }

    /**
     * The reference HITS scores, or null where the leading eigenvector is not
     * unique and the measure is therefore undefined.
     *
     * @return array{list<float>, list<float>}|null
     */
    public function expectedHits(): ?array
    {
        /** @var array{hits_hubs: list<float>|null, hits_authorities: list<float>|null} $expected */
        $expected = $this->expected;

        if ($expected['hits_hubs'] === null || $expected['hits_authorities'] === null) {
            return null;
        }

        return [$expected['hits_hubs'], $expected['hits_authorities']];
    }

    /** @return list<int> */
    public function groundTruth(): array
    {
        /** @var array{ground_truth?: list<int>} $expected */
        $expected = $this->expected;

        return $expected['ground_truth']
            ?? throw new RuntimeException("Fixture {$this->name} carries no ground truth");
    }

    /**
     * What leidenalg recovers of the ground truth, over many seeds. The bar
     * for this library is that band, not perfection: on a real graph nobody
     * recovers the truth exactly, and a test demanding it would only be
     * satisfiable by overfitting.
     *
     * @return array{nmi: array{min: float, max: float, mean: float}, ari: array{min: float, max: float, mean: float}, communities: array{min: int, max: int}, modularity: array{min: float, max: float}}
     */
    public function referenceScores(): array
    {
        /** @var array{reference?: array<string, mixed>} $expected */
        $expected = $this->expected;

        /** @var array{nmi: array{min: float, max: float, mean: float}, ari: array{min: float, max: float, mean: float}, communities: array{min: int, max: int}, modularity: array{min: float, max: float}} $reference */
        $reference = $expected['reference']
            ?? throw new RuntimeException("Fixture {$this->name} carries no reference scores");

        return $reference;
    }

    /** @return list<string> */
    public static function names(): array
    {
        return self::namesFrom('graph/index.json');
    }

    /** @return list<string> */
    private static function namesFrom(string $relative): array
    {
        $path = Paths::fixture($relative);
        $raw = file_get_contents($path);

        if ($raw === false) {
            throw new RuntimeException("Missing {$path}");
        }

        /** @var list<array{name: string}> $index */
        $index = json_decode($raw, associative: true, flags: JSON_THROW_ON_ERROR);

        return array_map(static fn (array $entry): string => $entry['name'], $index);
    }

    public function graph(): Graph
    {
        return Graph::undirected($this->nodes, $this->edges);
    }

    public function groundTruthPartition(): Partition
    {
        return Partition::fromMembership($this->groundTruth());
    }

    /** @return list<float> */
    public function expectedVector(string $statistic): array
    {
        /** @var array{centrality: array<string, list<float>>} $expected */
        $expected = $this->expected;

        return $expected['centrality'][$statistic]
            ?? throw new RuntimeException("Fixture {$this->name} has no expected '{$statistic}'");
    }

    /** @return array{min: float, max: float, mean: float, stdev: float} */
    public function leidenModularityEnvelope(): array
    {
        /** @var array{leiden: array{modularity_objective: array{modularity: array{min: float, max: float, mean: float, stdev: float}}}} $expected */
        $expected = $this->expected;

        return $expected['leiden']['modularity_objective']['modularity'];
    }

    /**
     * The CPM bands, keyed by the resolution they were measured at.
     *
     * Each records what leidenalg's CPM produced over fifty seeds: the
     * modularity of the partitions it found, and how many communities. The
     * modularity is incidental there rather than the objective, which is why
     * the test that reads this asks less of it than the modularity envelope
     * does.
     *
     * @return array<string, array{
     *     modularity: array{min: float, max: float, mean: float, stdev: float},
     *     communities: array{min: int, max: int},
     *     seeds: int
     * }>
     */
    public function leidenConstantPottsBands(): array
    {
        /** @var array{leiden: array{cpm: array<string, array{modularity: array{min: float, max: float, mean: float, stdev: float}, communities: array{min: int, max: int}, seeds: int}>}} $expected */
        $expected = $this->expected;

        return $expected['leiden']['cpm'];
    }

    /** @return array{min: int, max: int} */
    public function leidenCommunityCount(): array
    {
        /** @var array{leiden: array{modularity_objective: array{communities: array{min: int, max: int}}}} $expected */
        $expected = $this->expected;

        return $expected['leiden']['modularity_objective']['communities'];
    }
}
