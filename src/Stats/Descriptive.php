<?php

declare(strict_types=1);

namespace Vegoia\Stats;

use function array_is_list;
use function array_values;
use function count;
use function is_array;
use function iterator_to_array;
use function max;
use function min;
use function sort;
use function sqrt;

use Vegoia\Exception\InvalidArgument;
use Vegoia\Support\CompensatedSum;
use Vegoia\Support\ExactProduct;

/**
 * Univariate summary statistics, computed the accurate way.
 *
 * The textbook one-pass variance -- sum(x^2)/n - mean^2 -- is the single most
 * common numerical bug in statistical code. It subtracts two large, nearly
 * equal numbers, so when the data are large and tightly clustered every
 * significant digit cancels. On NIST's NumAcc1 (three 8-digit integers whose
 * standard deviation is exactly 1) it returns 0.
 *
 * This class uses the corrected two-pass algorithm of Chan, Golub & LeVeque
 * instead. The first pass finds the mean; the second accumulates deviations
 * *and* their plain sum, which would be zero in exact arithmetic and so
 * measures the error left in the mean. Subtracting that residual back out
 * recovers the digits the first pass lost. Both passes sum with Neumaier
 * compensation.
 *
 * Instances are immutable and memoise what they compute, so asking for the
 * mean, then the variance, then the skewness walks the data a bounded number
 * of times rather than once per question.
 *
 * @see T.F. Chan, G.H. Golub & R.J. LeVeque (1983), "Algorithms for Computing
 *      the Sample Variance: Analysis and Recommendations", The American
 *      Statistician 37(3), 242-247.
 */
final class Descriptive
{
    private ?float $mean = null;

    /** Sum of squared deviations about the mean, residual-corrected. */
    private ?float $sumSquaredDeviations = null;

    /** @var list<float>|null */
    private ?array $sorted = null;

    /** @param list<float> $values */
    private function __construct(
        private readonly array $values,
        private readonly Precision $precision = Precision::Extended,
    ) {
    }

    /**
     * @param iterable<float|int> $values
     * @param Precision           $precision Extended by default: accurate to
     *        the limit of the input, at roughly ten times the cost. See the
     *        enum for when Fast is the better trade.
     */
    public static function of(iterable $values, Precision $precision = Precision::Extended): self
    {
        if (! is_array($values)) {
            $values = iterator_to_array($values, preserve_keys: false);
        } elseif (! array_is_list($values)) {
            $values = array_values($values);
        }

        $floats = [];
        foreach ($values as $value) {
            $floats[] = (float) $value;
        }

        return new self($floats, $precision);
    }

    /** The same sample, computed the other way. */
    public function with(Precision $precision): self
    {
        return $precision === $this->precision ? $this : new self($this->values, $precision);
    }

    public function precision(): Precision
    {
        return $this->precision;
    }

    /** @return list<float> */
    public function values(): array
    {
        return $this->values;
    }

    public function count(): int
    {
        return count($this->values);
    }

    public function isEmpty(): bool
    {
        return $this->values === [];
    }

    public function sum(): float
    {
        if ($this->precision === Precision::Fast) {
            $total = 0.0;

            foreach ($this->values as $value) {
                $total += $value;
            }

            return $total;
        }

        return CompensatedSum::of($this->values);
    }

    public function mean(): float
    {
        if ($this->mean !== null) {
            return $this->mean;
        }

        $n = $this->count();

        if ($n === 0) {
            throw InvalidArgument::emptyDataset('the mean');
        }

        if ($this->precision === Precision::Fast) {
            return $this->mean = $this->sum() / $n;
        }

        // Divided inside the accumulator so the compensation survives the
        // division. Dividing sum() would round twice and lose up to two
        // digits on large, tightly clustered samples -- which then propagates
        // into every deviation computed from the mean. The division itself
        // adds about 4% to the cost of the compensated sum, so once the sum is
        // being compensated there is no reason not to.
        $accumulator = new CompensatedSum();

        foreach ($this->values as $value) {
            $accumulator->add($value);
        }

        return $this->mean = $accumulator->dividedBy((float) $n);
    }

    public function min(): float
    {
        if ($this->values === []) {
            throw InvalidArgument::emptyDataset('the minimum');
        }

        return min($this->values);
    }

    public function max(): float
    {
        if ($this->values === []) {
            throw InvalidArgument::emptyDataset('the maximum');
        }

        return max($this->values);
    }

    public function range(): float
    {
        return $this->max() - $this->min();
    }

    /** Sample variance, denominator n-1 (Bessel-corrected). */
    public function variance(): float
    {
        $n = $this->count();

        if ($n < 2) {
            throw InvalidArgument::tooFewValues('Sample variance', $n, 2);
        }

        return $this->centredSumOfSquares() / ($n - 1);
    }

    /** Population variance, denominator n. */
    public function populationVariance(): float
    {
        $n = $this->count();

        if ($n === 0) {
            throw InvalidArgument::emptyDataset('the population variance');
        }

        return $this->centredSumOfSquares() / $n;
    }

    public function stdDev(): float
    {
        return sqrt($this->variance());
    }

    public function populationStdDev(): float
    {
        return sqrt($this->populationVariance());
    }

    /** Standard error of the mean. */
    public function standardError(): float
    {
        return $this->stdDev() / sqrt((float) $this->count());
    }

    /** Coefficient of variation, the standard deviation relative to the mean. */
    public function coefficientOfVariation(): float
    {
        $mean = $this->mean();

        if ($mean === 0.0) {
            throw InvalidArgument::outOfRange('Coefficient of variation needs a non-zero mean; the mean', 0.0, PHP_FLOAT_MIN, PHP_FLOAT_MAX);
        }

        return $this->stdDev() / $mean;
    }

    /**
     * Lag-k autocorrelation, in the form NIST certifies it: deviations are
     * taken about the single overall mean and the denominator spans every
     * observation, not just the overlapping ones.
     */
    public function autocorrelation(int $lag = 1): float
    {
        $n = $this->count();

        if ($lag < 1) {
            throw InvalidArgument::outOfRange('Autocorrelation lag', (float) $lag, 1.0, (float) max(1, $n - 1));
        }

        if ($n <= $lag) {
            throw InvalidArgument::tooFewValues("Lag-{$lag} autocorrelation", $n, $lag + 1);
        }

        $mean = $this->mean();

        if ($this->precision === Precision::Fast) {
            $numerator = 0.0;
            $denominator = 0.0;

            for ($i = 0; $i + $lag < $n; $i++) {
                $numerator += ($this->values[$i] - $mean) * ($this->values[$i + $lag] - $mean);
            }

            foreach ($this->values as $value) {
                $deviation = $value - $mean;
                $denominator += $deviation * $deviation;
            }

            return $numerator / $denominator;
        }

        $numerator = new CompensatedSum();

        // Exact products, not merely a compensated sum of rounded ones. The
        // deviations here are small differences between large values, so each
        // product loses low bits that no later summation can recover -- worth
        // half a digit on real data, and five on NIST's accuracy sets.
        for ($i = 0; $i + $lag < $n; $i++) {
            ExactProduct::accumulate(
                $numerator,
                $this->values[$i] - $mean,
                $this->values[$i + $lag] - $mean,
            );
        }

        return $numerator->value() / $this->exactSumOfSquares();
    }

    /** Fisher-Pearson sample skewness (the g1 moment estimator). */
    public function skewness(): float
    {
        $n = $this->count();

        if ($n < 3) {
            throw InvalidArgument::tooFewValues('Skewness', $n, 3);
        }

        $mean = $this->mean();
        $third = new CompensatedSum();

        foreach ($this->values as $value) {
            $deviation = $value - $mean;
            $third->add($deviation * $deviation * $deviation);
        }

        $m2 = $this->centredSumOfSquares() / $n;
        $m3 = $third->value() / $n;

        return $m3 / ($m2 ** 1.5);
    }

    /** Excess kurtosis: 0 for a normal distribution, not 3. */
    public function kurtosis(): float
    {
        $n = $this->count();

        if ($n < 4) {
            throw InvalidArgument::tooFewValues('Kurtosis', $n, 4);
        }

        $mean = $this->mean();
        $fourth = new CompensatedSum();

        foreach ($this->values as $value) {
            $deviation = ($value - $mean) ** 2;
            $fourth->add($deviation * $deviation);
        }

        $m2 = $this->centredSumOfSquares() / $n;
        $m4 = $fourth->value() / $n;

        return $m4 / ($m2 * $m2) - 3.0;
    }

    public function median(): float
    {
        return $this->quantile(0.5);
    }

    /**
     * Linear-interpolation quantile: the R type-7 definition, which is also
     * numpy's default. Stated explicitly because the nine competing
     * definitions disagree, and silent disagreement is worse than either.
     */
    public function quantile(float $p): float
    {
        if ($p < 0.0 || $p > 1.0) {
            throw InvalidArgument::outOfRange('Quantile probability', $p, 0.0, 1.0);
        }

        $n = $this->count();

        if ($n === 0) {
            throw InvalidArgument::emptyDataset('a quantile');
        }

        $sorted = $this->sorted();
        $position = $p * ($n - 1);
        $lower = (int) $position;
        $fraction = $position - $lower;

        if ($fraction === 0.0 || $lower + 1 >= $n) {
            return $sorted[$lower];
        }

        return $sorted[$lower] + $fraction * ($sorted[$lower + 1] - $sorted[$lower]);
    }

    /** Interquartile range, Q3 - Q1. */
    public function iqr(): float
    {
        return $this->quantile(0.75) - $this->quantile(0.25);
    }

    /**
     * The corrected two-pass sum of squared deviations.
     *
     * `$residual` is sum(x_i - mean), which exact arithmetic would make zero.
     * Whatever it actually holds is the rounding error left in the mean, and
     * subtracting residual^2 / n removes that error's contribution to the sum
     * of squares. This one term is the difference between roughly 8 correct
     * digits and 15 on NIST's NumAcc datasets.
     */
    private function centredSumOfSquares(): float
    {
        if ($this->sumSquaredDeviations !== null) {
            return $this->sumSquaredDeviations;
        }

        $mean = $this->mean();

        if ($this->precision === Precision::Fast) {
            $plain = 0.0;

            foreach ($this->values as $value) {
                $deviation = $value - $mean;
                $plain += $deviation * $deviation;
            }

            return $this->sumSquaredDeviations = $plain;
        }

        $squares = new CompensatedSum();
        $residual = new CompensatedSum();

        foreach ($this->values as $value) {
            $deviation = $value - $mean;
            $squares->add($deviation * $deviation);
            $residual->add($deviation);
        }

        $correction = $residual->value() ** 2 / $this->count();

        return $this->sumSquaredDeviations = $squares->value() - $correction;
    }

    /**
     * Sum of squared deviations with exact products.
     *
     * The autocorrelation's denominator, kept separate from
     * centredSumOfSquares() because that one carries the residual correction
     * the variance needs, while this one needs the extra precision instead --
     * NIST defines r(k) against the plain sum of squares.
     */
    private function exactSumOfSquares(): float
    {
        $mean = $this->mean();
        $squares = new CompensatedSum();

        foreach ($this->values as $value) {
            $deviation = $value - $mean;
            ExactProduct::accumulate($squares, $deviation, $deviation);
        }

        return $squares->value();
    }

    /** @return list<float> */
    private function sorted(): array
    {
        if ($this->sorted !== null) {
            return $this->sorted;
        }

        $sorted = $this->values;
        sort($sorted);

        return $this->sorted = $sorted;
    }
}
