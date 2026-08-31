<?php

declare(strict_types=1);

namespace Vegoia\Tests\Support;

use function array_key_exists;
use function file_get_contents;
use function is_array;
use function json_decode;

use const JSON_THROW_ON_ERROR;

use function min;

use RuntimeException;

/**
 * The accuracy ceiling imposed by binary64, measured rather than assumed.
 *
 * Three of the NIST certified values cannot be reproduced to full precision
 * by anyone: NumAcc4's observations are not representable as doubles, so the
 * error is already baked into the input before a single operation runs. If
 * the suite simply demanded 13 digits everywhere it would fail forever, and
 * the tempting fix -- lowering the constant until green -- destroys the whole
 * point of testing against certified values.
 *
 * So the ceilings come from `tools/generate_nist_attainable.py`, which
 * measures what numpy reaches on the same data. Vegoia is then required to
 * come within `MARGIN` digits of an independent implementation. That keeps
 * the bar tied to the arithmetic: if someone reintroduces a naive variance,
 * the LRE collapses by several digits and the test still fails.
 */
final class AttainableAccuracy
{
    /**
     * Slack for a different but equally valid order of operations. Half a
     * digit is a factor of ~3 in the error, far below the several orders of
     * magnitude a genuinely worse algorithm gives up.
     */
    public const float MARGIN = 0.5;

    /** @var array<string, array<string, array<string, float|null>>> keyed by source file */
    private static array $ceilings = [];

    /**
     * Digits to require for one statistic: the standard bar, unless the
     * arithmetic itself cannot get there.
     */
    public static function required(
        string $dataset,
        string $statistic,
        string $source = 'nist/attainable.json',
    ): float {
        $measured = self::ceilings($source);
        $baseline = (float) Lre::DEFAULT_DIGITS;
        $margin = self::MARGIN;

        // Deliberately not `??`: a null value here means "hit exactly", which
        // is the most interesting case, and `??` would silently treat it as a
        // missing key.
        if (! isset($measured[$dataset]) || ! array_key_exists($statistic, $measured[$dataset])) {
            throw new RuntimeException("No measured ceiling for {$dataset}.{$statistic}");
        }

        $ceiling = $measured[$dataset][$statistic];

        // null means numpy hit the certified value exactly, so nothing caps us.
        if ($ceiling === null) {
            return $baseline;
        }

        return min($baseline, $ceiling - $margin);
    }

    /** @return array<string, array<string, float|null>> */
    private static function ceilings(string $source): array
    {
        if (isset(self::$ceilings[$source])) {
            return self::$ceilings[$source];
        }

        $path = Paths::fixture($source);
        $raw = file_get_contents($path);

        if ($raw === false) {
            throw new RuntimeException("Missing {$path}. Regenerate it with the matching script in tools/.");
        }

        $decoded = json_decode($raw, associative: true, flags: JSON_THROW_ON_ERROR);

        if (! is_array($decoded) || ! isset($decoded['datasets']) || ! is_array($decoded['datasets'])) {
            throw new RuntimeException("Malformed {$path}: no 'datasets' map.");
        }

        /** @var array<string, array<string, float|null>> $datasets */
        $datasets = $decoded['datasets'];

        return self::$ceilings[$source] = $datasets;
    }
}
