<?php

declare(strict_types=1);

namespace Vegoia\Support;

use function abs;

use Vegoia\Exception\InvalidArgument;

/**
 * Neumaier summation: a running sum that keeps the bits floating-point
 * addition would otherwise drop.
 *
 * Adding n doubles naively lets rounding error grow with n, and it grows
 * fastest exactly where it hurts -- when a large running total swallows small
 * addends, or when near-equal magnitudes cancel. Neumaier's variant of Kahan
 * summation carries that lost low-order part in a separate accumulator and
 * folds it back in at the end, which bounds the error independently of n.
 *
 * It costs roughly four flops per element instead of one. For this library
 * that is the right trade: the NIST NumAcc datasets exist precisely to fail
 * implementations that took the cheaper path.
 *
 * @see A. Neumaier (1974), "Rundungsfehleranalyse einiger Verfahren zur
 *      Summation endlicher Summen", ZAMM 54, 39-51.
 */
final class CompensatedSum
{
    private float $sum = 0.0;

    /** The low-order bits addition discarded, accumulated separately. */
    private float $compensation = 0.0;

    public function add(float $value): self
    {
        $t = $this->sum + $value;

        // Whichever operand is larger keeps its bits; the other one loses them,
        // so recover the loss from the smaller side.
        $this->compensation += abs($this->sum) >= abs($value)
            ? ($this->sum - $t) + $value
            : ($value - $t) + $this->sum;

        $this->sum = $t;

        return $this;
    }

    public function value(): float
    {
        return $this->sum + $this->compensation;
    }

    /**
     * exp() of the accumulated total, without collapsing it first.
     *
     * The same reasoning as dividedBy(). A logarithm around -673, which is
     * what the tail of the error function needs, has an ulp of 1.1e-13, so
     * folding the compensation into it before exponentiating throws away
     * three digits of the answer -- measured, erfc(26) came out to 12.81
     * digits that way against SciPy's 15.35, and the reference test failed.
     *
     * exp(a + b) = exp(a) exp(b) exactly, and here b is the compensation:
     * small enough that exp(b) is computed to full relative accuracy, so the
     * product recovers what the collapse would have discarded.
     */
    public function exponentiated(): float
    {
        return exp($this->sum) * exp($this->compensation);
    }

    /**
     * Divide the accumulated total, keeping the compensation through the
     * division.
     *
     * value() collapses the head and the compensation into one double before
     * returning, so dividing its result rounds twice: once folding the tail
     * in, once dividing. Dividing here instead recovers the remainder exactly
     * -- via the error term of q * divisor -- and folds it back, so only the
     * final result rounds.
     *
     * It is worth two digits on a mean of tightly clustered large values, and
     * it is what lets NIST's NumAcc3 and NumAcc4 come out exact. GSL reaches
     * the same place by accumulating in long double, which PHP does not have.
     */
    /**
     * $value minus the accumulated total, without collapsing it first.
     *
     * The same reasoning as exponentiated() and dividedBy(), and it pays
     * where a residual is formed: subtracting a fitted value from an observed
     * one is a cancellation, and cancellation is precisely when the discarded
     * compensation stops being negligible.
     *
     * Collapsing first computes value - (sum + compensation), which rounds the
     * addition and then subtracts. Subtracting the head first does better than
     * avoid one rounding: when value and sum are within a factor of two, which
     * is what a well-fitted point means, Sterbenz's lemma makes value - sum
     * exact, so the whole of the compensation survives into the answer instead
     * of being rounded away against a number a million times its size.
     *
     * Measured on the NIST least squares sets, where the residual sum of
     * squares is the quantity that suffers. It is worth 0.17 of a digit on
     * Pontius, 12.96 to 13.13, which is the set where the cancellation is
     * worst; 0.06 on Norris; and nothing at all on the rest, whose residuals
     * are not small enough against the response for the compensation to be
     * reachable. The overall F follows, being that sum twice over.
     */
    public function subtractedFrom(float $value): float
    {
        return ($value - $this->sum) - $this->compensation;
    }

    public function dividedBy(float $divisor): float
    {
        if ($divisor === 0.0) {
            throw InvalidArgument::outOfRange('Divisor', 0.0, PHP_FLOAT_MIN, INF);
        }

        $quotient = $this->sum / $divisor;

        [$product, $productError] = ExactProduct::of($quotient, $divisor);

        // What the head and the compensation still owe, after removing what
        // the quotient accounts for.
        [$remainder, $carry] = self::twoSum(
            $this->sum - $product,
            $this->compensation - $productError,
        );

        return $quotient + ($remainder + $carry) / $divisor;
    }

    /**
     * The sum of two doubles and the bits it lost, exactly.
     *
     * @return array{float, float}
     */
    public static function twoSum(float $a, float $b): array
    {
        $sum = $a + $b;
        $shifted = $sum - $a;

        return [$sum, ($a - ($sum - $shifted)) + ($b - $shifted)];
    }

    /** @param iterable<float|int> $values */
    public static function of(iterable $values): float
    {
        $accumulator = new self();

        foreach ($values as $value) {
            $accumulator->add((float) $value);
        }

        return $accumulator->value();
    }
}
