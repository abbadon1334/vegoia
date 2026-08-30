<?php

declare(strict_types=1);

namespace Vegoia\Tests\Support;

use function array_slice;
use function array_values;
use function count;
use function file;
use function implode;
use function preg_match;
use function preg_split;

use RuntimeException;

use function sprintf;
use function str_starts_with;
use function trim;

/**
 * A NIST StRD one-way ANOVA dataset.
 *
 * One quirk in the source data is worked around here rather than silently
 * absorbed: AtmWtAg.dat declares its certified values on lines 41-47 and then
 * writes the residual standard deviation on line 48. Reading only the declared
 * range loses that value on that one dataset, so the residual line is located
 * over the whole file.
 *
 * @see https://www.itl.nist.gov/div898/strd/anova/anova.html
 */
final readonly class NistAnova
{
    /**
     * @param list<list<float>> $groups
     */
    private function __construct(
        public string $name,
        public array $groups,
        public float $betweenSumOfSquares,
        public float $withinSumOfSquares,
        public int $betweenDegreesOfFreedom,
        public int $withinDegreesOfFreedom,
        public float $fStatistic,
        public float $rSquared,
        public float $residualStandardDeviation,
    ) {
    }

    public static function load(string $name): self
    {
        $path = sprintf('%s/resources/fixtures/nist/anova/%s.dat', Paths::root(), $name);
        $lines = file($path, FILE_IGNORE_NEW_LINES);

        if ($lines === false) {
            throw new RuntimeException("Missing NIST fixture: {$path}");
        }

        $head = implode("\n", array_slice($lines, 0, 20));

        if (preg_match('/Certified Values\s*\(lines\s+(\d+)\s+to\s+(\d+)\)/', $head, $m) !== 1) {
            throw new RuntimeException("No certified span declared in {$path}");
        }
        [$certifiedFrom, $certifiedTo] = [(int) $m[1], (int) $m[2]];

        if (preg_match('/Data\s*\(lines\s+(\d+)\s+to\s+(\d+)\)/', $head, $m) !== 1) {
            throw new RuntimeException("No data span declared in {$path}");
        }
        [$dataFrom, $dataTo] = [(int) $m[1], (int) $m[2]];

        /** @var array<string, list<float>> $byLabel */
        $byLabel = [];

        foreach (array_slice($lines, $dataFrom - 1, $dataTo - $dataFrom + 1) as $line) {
            $fields = preg_split('/\s+/', trim($line), flags: PREG_SPLIT_NO_EMPTY);

            if ($fields === false || count($fields) < 2) {
                continue;
            }

            $byLabel[$fields[0]][] = (float) $fields[1];
        }

        $between = null;
        $within = null;
        $rSquared = null;

        foreach (array_slice($lines, $certifiedFrom - 1, $certifiedTo - $certifiedFrom + 1) as $line) {
            $trimmed = trim($line);

            if (preg_match('/^(Between|Within)\s+\S+\s+(\d+)\s+(\S+)\s+(\S+)(?:\s+(\S+))?/', $trimmed, $m) === 1) {
                $row = ['df' => (int) $m[2], 'ss' => (float) $m[3], 'f' => isset($m[5]) ? (float) $m[5] : null];

                if ($m[1] === 'Between') {
                    $between = $row;
                } else {
                    $within = $row;
                }
            }

            if (str_starts_with($trimmed, 'Certified R-Squared') || str_starts_with($trimmed, 'R-Squared')) {
                $rSquared = self::trailingNumber($trimmed, $path);
            }
        }

        // Deliberately over the whole file: see the class docblock.
        $residual = null;

        foreach ($lines as $line) {
            if (str_starts_with(trim($line), 'Standard Deviation')) {
                $residual = self::trailingNumber(trim($line), $path);

                break;
            }
        }

        if ($between === null || $within === null || $rSquared === null || $residual === null) {
            throw new RuntimeException("Incomplete certified block in {$path}");
        }

        return new self(
            $name,
            array_values($byLabel),
            $between['ss'],
            $within['ss'],
            $between['df'],
            $within['df'],
            $between['f'] ?? 0.0,
            $rSquared,
            $residual,
        );
    }

    private static function trailingNumber(string $line, string $path): float
    {
        if (preg_match('/(-?\d+\.?\d*(?:[Ee][-+]?\d+)?)\s*$/', $line, $m) !== 1) {
            throw new RuntimeException("Cannot read a number from '{$line}' in {$path}");
        }

        return (float) $m[1];
    }
}
