<?php

declare(strict_types=1);

namespace Vegoia\Tests\Support;

use function array_slice;
use function count;
use function file;
use function is_file;
use function preg_match;

use RuntimeException;

use function sprintf;
use function str_replace;
use function trim;

/**
 * A NIST StRD univariate summary-statistics dataset.
 *
 * The .dat files are self-describing: the header states on which lines the
 * certified values and the data live, so the parser reads those positions
 * rather than pattern-matching the whole file. Nothing here is transcribed by
 * hand -- the certified constants are read from the shipped file, so a typo
 * in a test is not possible.
 *
 * @see https://www.itl.nist.gov/div898/strd/univ/homepage.html
 */
final readonly class NistUnivariate
{
    /** @param list<float> $values */
    private function __construct(
        public string $name,
        public float $mean,
        public float $stdDev,
        public float $autocorrelation,
        public array $values,
    ) {
    }

    public static function load(string $name): self
    {
        $path = sprintf('%s/resources/fixtures/nist/univ/%s.dat', Paths::root(), $name);

        if (! is_file($path)) {
            throw new RuntimeException("Missing NIST fixture: {$path}");
        }

        $lines = file($path, FILE_IGNORE_NEW_LINES);

        if ($lines === false) {
            throw new RuntimeException("Unreadable NIST fixture: {$path}");
        }

        [$certifiedFrom, $certifiedTo] = self::span($lines, 'Certified Values', $path);
        [$dataFrom, $dataTo] = self::span($lines, 'Data', $path);

        $certified = [];
        foreach (array_slice($lines, $certifiedFrom - 1, $certifiedTo - $certifiedFrom + 1) as $line) {
            $certified[] = self::number($line, $path);
        }

        if (count($certified) !== 3) {
            throw new RuntimeException("Expected 3 certified values in {$path}, got " . count($certified));
        }

        $values = [];
        foreach (array_slice($lines, $dataFrom - 1, $dataTo - $dataFrom + 1) as $line) {
            $values[] = self::number($line, $path);
        }

        return new self($name, $certified[0], $certified[1], $certified[2], $values);
    }

    /**
     * Header lines look like:
     *   Certified Values: lines 41 to  43     (=   3)
     *   Data            : lines 61 to 278     (= 218)
     *
     * @param  list<string> $lines
     * @return array{int, int}
     */
    private static function span(array $lines, string $label, string $path): array
    {
        foreach ($lines as $line) {
            if (preg_match('/^\s*' . $label . '\s*:\s*lines\s+(\d+)\s+to\s+(\d+)/', $line, $m) === 1) {
                return [(int) $m[1], (int) $m[2]];
            }
        }

        throw new RuntimeException("No '{$label}' span declared in {$path}");
    }

    /**
     * Certified lines carry a label and sometimes an "(exact)" marker:
     *   Sample Standard Deviation (denom. = n-1)      s:         1 (exact)
     * Data lines carry the number alone. Taking the last float-looking token
     * before any marker handles both without special cases.
     */
    private static function number(string $line, string $path): float
    {
        $clean = trim(str_replace('(exact)', '', $line));

        if (preg_match('/(-?\d+\.?\d*(?:[Ee][-+]?\d+)?)\s*$/', $clean, $m) !== 1) {
            throw new RuntimeException("Cannot read a number from '{$line}' in {$path}");
        }

        return (float) $m[1];
    }
}
