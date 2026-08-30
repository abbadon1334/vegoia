<?php

declare(strict_types=1);

namespace Vegoia\Tests\Support;

use function array_slice;
use function count;
use function file;
use function implode;
use function in_array;
use function preg_match;
use function preg_split;

use RuntimeException;

use function sprintf;
use function str_starts_with;
use function trim;

/**
 * A NIST StRD linear least squares dataset.
 *
 * Two traps live in these files. The parameter count cannot be read off the
 * model line, because Filip writes it as "B0 + B1*x + B2*(x**2) + ... +
 * B10*(x**10)" and the literal ellipsis hides six terms -- so the count comes
 * from the certified block, which lists every one. And a dataset with a single
 * predictor but more than two parameters is polynomial in that predictor
 * rather than linear in several, which changes the design matrix entirely.
 *
 * @see https://www.itl.nist.gov/div898/strd/lls/lls.shtml
 */
final readonly class NistRegression
{
    /**
     * @param list<float>       $estimates certified coefficients, B0 first
     * @param list<float>       $errors    certified standard errors
     * @param list<float>       $response
     * @param list<list<float>> $predictors one row per observation
     */
    private function __construct(
        public string $name,
        public bool $hasIntercept,
        public int $parameters,
        public int $predictorCount,
        public array $estimates,
        public array $errors,
        public float $residualStandardDeviation,
        public float $rSquared,
        public array $response,
        public array $predictors,
    ) {
    }

    /** Polynomial in the single predictor rather than linear in several. */
    public function isPolynomial(): bool
    {
        return $this->predictorCount === 1 && $this->parameters > 2;
    }

    public function degree(): int
    {
        return $this->parameters - ($this->hasIntercept ? 1 : 0);
    }

    public static function load(string $name): self
    {
        $path = sprintf('%s/resources/fixtures/nist/lls/%s.dat', Paths::root(), $name);
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

        if (preg_match('/(\d+)\s+Predictor Variable/', $head, $m) !== 1) {
            throw new RuntimeException("No predictor count declared in {$path}");
        }
        $predictorCount = (int) $m[1];

        $indices = [];
        $estimates = [];
        $errors = [];
        $residualStandardDeviation = null;
        $rSquared = null;

        foreach (array_slice($lines, $certifiedFrom - 1, $certifiedTo - $certifiedFrom + 1) as $line) {
            $trimmed = trim($line);

            if (preg_match('/^B(\d+)\s+(\S+)\s+(\S+)/', $trimmed, $m) === 1) {
                $indices[] = (int) $m[1];
                $estimates[] = (float) $m[2];
                $errors[] = (float) $m[3];

                continue;
            }

            if ($residualStandardDeviation === null && str_starts_with($trimmed, 'Standard Deviation')) {
                $residualStandardDeviation = self::trailingNumber($trimmed, $path);
            }

            if (str_starts_with($trimmed, 'R-Squared')) {
                $rSquared = self::trailingNumber($trimmed, $path);
            }
        }

        if ($estimates === [] || $residualStandardDeviation === null || $rSquared === null) {
            throw new RuntimeException("Incomplete certified block in {$path}");
        }

        $response = [];
        $predictors = [];

        foreach (array_slice($lines, $dataFrom - 1, $dataTo - $dataFrom + 1) as $line) {
            $fields = preg_split('/\s+/', trim($line), flags: PREG_SPLIT_NO_EMPTY);

            if ($fields === false || $fields === []) {
                continue;
            }

            $response[] = (float) $fields[0];
            $row = [];

            for ($i = 1; $i < count($fields); $i++) {
                $row[] = (float) $fields[$i];
            }

            $predictors[] = $row;
        }

        return new self(
            $name,
            in_array(0, $indices, true),
            count($indices),
            $predictorCount,
            $estimates,
            $errors,
            $residualStandardDeviation,
            $rSquared,
            $response,
            $predictors,
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
