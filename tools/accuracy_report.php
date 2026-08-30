<?php

declare(strict_types=1);

/**
 * Report how many correct significant digits Vegoia actually achieves on the
 * NIST Statistical Reference Datasets.
 *
 * The test suite asserts a floor; this prints the real figure, so a claim in
 * the README can be checked rather than believed.
 *
 * Usage: php tools/accuracy_report.php
 */

use Vegoia\Stats\Descriptive;
use Vegoia\Stats\Regression\LeastSquares;
use Vegoia\Tests\Support\Lre;
use Vegoia\Tests\Support\NistRegression;
use Vegoia\Tests\Support\NistUnivariate;

require __DIR__ . '/../vendor/autoload.php';

function digits(float $computed, float $certified): string
{
    $lre = Lre::of($computed, $certified);

    return is_finite($lre) ? sprintf('%5.2f', $lre) : ' exact';
}

echo "Univariate summary statistics (correct significant digits)\n";
printf("%-10s %8s %8s %8s\n", 'dataset', 'mean', 'stdDev', 'r(1)');
echo str_repeat('-', 38), "\n";

foreach (['PiDigits', 'Lottery', 'Lew', 'Mavro', 'Michelso',
          'NumAcc1', 'NumAcc2', 'NumAcc3', 'NumAcc4'] as $name) {
    $set = NistUnivariate::load($name);
    $stats = Descriptive::of($set->values);

    printf(
        "%-10s %8s %8s %8s\n",
        $name,
        digits($stats->mean(), $set->mean),
        digits($stats->stdDev(), $set->stdDev),
        digits($stats->autocorrelation(1), $set->autocorrelation),
    );
}

echo "\nLinear least squares (worst case across parameters)\n";
printf("%-10s %4s %8s %8s %8s\n", 'dataset', 'p', 'coef', 'stdErr', 'residSD');
echo str_repeat('-', 42), "\n";

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

    printf(
        "%-10s %4d %8s %8s %8s\n",
        $name,
        $fit->parameters,
        is_finite($worstCoefficient) ? sprintf('%5.2f', $worstCoefficient) : ' exact',
        is_finite($worstError) ? sprintf('%5.2f', $worstError) : ' exact',
        digits($fit->residualStandardDeviation, $set->residualStandardDeviation),
    );
}

echo "\nA double carries ~15.95 decimal digits. NIST treats 11 as adequate for a\n";
echo "statistical package; where a figure is lower, the input itself cannot be\n";
echo "represented more precisely (see resources/fixtures/nist/attainable.json).\n";
