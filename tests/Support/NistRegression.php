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
        public int $regressionDegreesOfFreedom,
        public int $residualDegreesOfFreedom,
        public float $regressionSumOfSquares,
        public float $residualSumOfSquares,
        public float $fStatistic,
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

            // The certified analysis of variance, which was being read past.
            // Its F statistic is the one number in these files that exercises
            // the whole fit at once -- coefficients, residual sum of squares
            // and both degrees of freedom together -- so leaving it unparsed
            // meant the overall test had no certified reference at all.
            //
            // Regression carries a fourth column, Residual only three:
            //
            //   Regression    1   4255954.13232369  4255954.13232369  5436385.54079785
            //   Residual     34   26.6173985294224  0.782864662630069
            if (preg_match('/^Regression\s+(\d+)\s+(\S+)\s+\S+\s+(\S+)/', $trimmed, $m) === 1) {
                $regressionDegreesOfFreedom = (int) $m[1];
                $regressionSumOfSquares = self::number($m[2], $path);
                // Wampler1 and Wampler2 fit exactly, so NIST writes the word
                // rather than a number. Kept as INF, which is the true value.
                $fStatistic = $m[3] === 'Infinity' ? INF : self::number($m[3], $path);
            }

            if (preg_match('/^Residual\s+(\d+)\s+(\S+)\s+(\S+)/', $trimmed, $m) === 1) {
                $residualDegreesOfFreedom = (int) $m[1];
                $residualSumOfSquares = self::number($m[2], $path);
            }
        }

        if ($estimates === [] || $residualStandardDeviation === null || $rSquared === null) {
            throw new RuntimeException("Incomplete certified block in {$path}");
        }

        if (
            ! isset(
                $regressionDegreesOfFreedom,
                $residualDegreesOfFreedom,
                $regressionSumOfSquares,
                $residualSumOfSquares,
                $fStatistic,
            )
        ) {
            throw new RuntimeException("No certified analysis of variance in {$path}");
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
            $regressionDegreesOfFreedom,
            $residualDegreesOfFreedom,
            $regressionSumOfSquares,
            $residualSumOfSquares,
            $fStatistic,
            $response,
            $predictors,
        );
    }

    /**
     * A single certified field.
     *
     * Separate from trailingNumber() because these are read from the middle
     * of a row and must be rejected if malformed, where a trailing match on a
     * partly-read line would silently take whatever digits it found.
     */
    private static function number(string $field, string $path): float
    {
        if (preg_match('/^-?\d+\.?\d*(?:[Ee][-+]?\d+)?$/', $field) !== 1) {
            throw new RuntimeException("Cannot read a number from '{$field}' in {$path}");
        }

        return (float) $field;
    }

    private static function trailingNumber(string $line, string $path): float
    {
        if (preg_match('/(-?\d+\.?\d*(?:[Ee][-+]?\d+)?)\s*$/', $line, $m) !== 1) {
            throw new RuntimeException("Cannot read a number from '{$line}' in {$path}");
        }

        return (float) $m[1];
    }
}
