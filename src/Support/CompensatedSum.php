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
