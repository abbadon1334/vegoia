<?php

declare(strict_types=1);

/**
 * Report how many correct significant digits Vegoia achieves on the NIST
 * Statistical Reference Datasets, beside what numpy and SciPy achieve.
 *
 * The test suite asserts a floor; this prints the real figure, so a claim in
 * the README can be checked rather than believed. Both columns are measured
 * against the same certified values, and the reference column is not
 * recomputed here -- it is read from the fixtures the generators wrote, so
 * the number in the table is the number the tests are held to.
 *
 * Usage: php tools/accuracy_report.php [--markdown]
 */

use Vegoia\Stats\Descriptive;
use Vegoia\Stats\Regression\LeastSquares;
use Vegoia\Tests\Support\Lre;
use Vegoia\Tests\Support\NistRegression;
use Vegoia\Tests\Support\NistUnivariate;

require __DIR__ . '/../vendor/autoload.php';

function digits(float $computed, float $certified): string
{
    return format(Lre::of($computed, $certified));
}

function format(float $lre): string
{
    return is_finite($lre) ? sprintf('%5.2f', $lre) : ' exact';
}

/**
 * What the reference reached, as the generators measured it.
 *
 * null in the fixture means it hit the certified value exactly, which is a
 * ceiling of everything rather than of nothing -- printing it as a missing
 * measurement would read as a failure.
 */
function reference(string $source, string $dataset, string $statistic): string
{
    static $cache = [];

    if (! isset($cache[$source])) {
        /** @var array{datasets: array<string, array<string, float|null>>} $decoded */
        $decoded = json_decode(
            (string) file_get_contents(__DIR__ . "/../resources/fixtures/nist/{$source}"),
            true,
            flags: JSON_THROW_ON_ERROR,
        );
        $cache[$source] = $decoded['datasets'];
    }

    if (! array_key_exists($statistic, $cache[$source][$dataset] ?? [])) {
        return '     -';
    }

    $measured = $cache[$source][$dataset][$statistic];

    return $measured === null ? ' exact' : sprintf('%5.2f', $measured);
}

/** The wider of the two, so a reader can see at a glance who won the row. */
function verdict(string $ours, string $theirs): string
{
    if ($ours === $theirs) {
        return ' =';
    }

    $a = trim($ours) === 'exact' ? INF : (float) $ours;
    $b = trim($theirs) === 'exact' ? INF : (float) $theirs;

    if ($a === INF && $b === INF) {
        return ' =';
    }

    return $a > $b ? ' +' : ($a < $b ? ' -' : ' =');
}

echo "Univariate summary statistics -- correct significant digits\n";
printf("%-10s %-16s %-16s %-16s\n", '', 'mean', 'stdDev', 'r(1)');
printf("%-10s %-16s %-16s %-16s\n", 'dataset', 'Vegoia  numpy', 'Vegoia  numpy', 'Vegoia  numpy');
echo str_repeat('-', 61), "\n";

foreach (['PiDigits', 'Lottery', 'Lew', 'Mavro', 'Michelso',
          'NumAcc1', 'NumAcc2', 'NumAcc3', 'NumAcc4'] as $name) {
    $set = NistUnivariate::load($name);
    $stats = Descriptive::of($set->values);

    $columns = [
        ['mean', digits($stats->mean(), $set->mean)],
        ['stdDev', digits($stats->stdDev(), $set->stdDev)],
        ['autocorrelation', digits($stats->autocorrelation(1), $set->autocorrelation)],
    ];

    printf('%-10s', $name);

    foreach ($columns as [$statistic, $ours]) {
        $theirs = reference('attainable.json', $name, $statistic);
        printf(' %6s %6s %-2s', trim($ours), trim($theirs), verdict($ours, $theirs));
    }

    echo "\n";
}

echo "\nLinear least squares -- worst case across the parameters of each fit\n";
printf("%-10s %4s %-16s %-16s %-16s\n", '', '', 'coefficients', 'std errors', 'residual SD');
printf("%-10s %4s %-16s %-16s %-16s\n", 'dataset', 'p', 'Vegoia  numpy', 'Vegoia  numpy', 'Vegoia  numpy');
echo str_repeat('-', 66), "\n";

foreach (['Norris', 'NoInt1', 'NoInt2', 'Pontius', 'Longley',
          'Wampler1', 'Wampler2', 'Wampler3', 'Wampler4', 'Wampler5', 'Filip'] as $name) {
    $set = NistRegression::load($name);

    if ($set->isPolynomial()) {
        $x = [];
        foreach ($set->predictors as $row) {
            $x[] = $row[0];
        }
        $fit = LeastSquares::polynomial($x, $set->response, $set->degree(), $set->hasIntercept);
    } else {
        $fit = LeastSquares::fit($set->predictors, $set->response, $set->hasIntercept);
    }

    $worstCoefficient = INF;
    $worstError = INF;

    foreach ($set->estimates as $i => $certified) {
        $worstCoefficient = min($worstCoefficient, Lre::of($fit->coefficients[$i], $certified));
    }

    foreach ($set->errors as $i => $certified) {
        $worstError = min($worstError, Lre::of($fit->standardErrors[$i], $certified));
    }

    $columns = [
        ['coefficients', format($worstCoefficient)],
        ['standardErrors', format($worstError)],
        ['residualStandardDeviation', digits($fit->residualStandardDeviation, $set->residualStandardDeviation)],
    ];

    printf('%-10s %4d', $name, $fit->parameters);

    foreach ($columns as [$statistic, $ours]) {
        $theirs = reference('attainable_lls.json', $name, $statistic);
        printf(' %6s %6s %-2s', trim($ours), trim($theirs), verdict($ours, $theirs));
    }

    echo "\n";
}

echo "\n+ Vegoia is more accurate on this row, - less, = the same.\n";
echo "\nA double carries ~15.95 decimal digits. NIST treats 11 as adequate for a\n";
echo "statistical package. The reference column is numpy solving the same way --\n";
echo "Householder QR, not SVD -- so a gap is a gap in the implementation and not\n";
echo "in the choice of algorithm (resources/fixtures/nist/attainable*.json).\n";
