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
        $path = Paths::fixture("graph/{$name}.json");
        $raw = file_get_contents($path);

        if ($raw === false) {
            throw new RuntimeException(
                "Missing graph fixture {$path}. Run: python3 tools/generate_graph_fixtures.py"
            );
        }

        $data = json_decode($raw, associative: true, flags: JSON_THROW_ON_ERROR);

        if (! is_array($data)) {
            throw new RuntimeException("Malformed graph fixture {$path}");
        }

        /** @var array{name: string, note: string, nodes: int, edges: list<array{int, int, float}>, expected: array<string, mixed>} $data */
        return new self($data['name'], $data['note'], $data['nodes'], $data['edges'], $data['expected']);
    }

    /** @return list<string> */
    public static function names(): array
    {
        $path = Paths::fixture('graph/index.json');
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

    /** @return array{min: int, max: int} */
    public function leidenCommunityCount(): array
    {
        /** @var array{leiden: array{modularity_objective: array{communities: array{min: int, max: int}}}} $expected */
        $expected = $this->expected;

        return $expected['leiden']['modularity_objective']['communities'];
    }
}
