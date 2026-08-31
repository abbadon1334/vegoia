<?php

declare(strict_types=1);

namespace Vegoia\Support;

/**
 * The product of two doubles, exactly: a head and the bits it discarded.
 *
 * Multiplying two doubles gives a result rounded to 53 bits, and the part
 * dropped is not small in the way a rounding error usually is -- it is up to
 * half an ulp of a product that may itself be huge. Summing many such products
 * accumulates those losses, and a compensated sum cannot recover them because
 * they were gone before the sum ever saw them.
 *
 * Dekker's algorithm splits each operand into two halves that multiply
 * exactly, so the four partial products reconstruct what the single rounded
 * multiplication threw away. Feeding both the head and the error into a
 * compensated sum then gives a dot product accurate to nearly twice the
 * working precision.
 *
 * This is what GSL achieves for free by accumulating in long double, which
 * carries 64 bits of mantissa on x86. PHP has no long double, so the precision
 * has to be built rather than borrowed.
 *
 * Costs about six flops per product instead of one, so it is used where the
 * accuracy is worth it -- lag autocorrelation, where it is worth half a digit
 * on real data and more on the NIST accuracy sets -- and not in the graph
 * kernels, where the inner loop runs millions of times.
 *
 * @see T.J. Dekker (1971), "A floating-point technique for extending the
 *      available precision", Numerische Mathematik 18, 224-242.
 */
final class ExactProduct
{
    /**
     * Splitting constant, 2^27 + 1.
     *
     * Multiplying by it and subtracting isolates the high 26 bits of a double's
     * mantissa; what remains is the low 27. Each half then fits in 27 bits, so
     * products of halves are exact.
     */
    private const float SPLIT = 134217729.0;

    /**
     * @return array{float, float} the rounded product, and the error such that
     *         product + error is the exact result
     */
    public static function of(float $a, float $b): array
    {
        $product = $a * $b;

        $ca = self::SPLIT * $a;
        $highA = $ca - ($ca - $a);
        $lowA = $a - $highA;

        $cb = self::SPLIT * $b;
        $highB = $cb - ($cb - $b);
        $lowB = $b - $highB;

        // Reassemble the exact product from parts that multiply without
        // rounding, then subtract what the single multiplication returned.
        $error = ((($highA * $highB - $product) + $highA * $lowB) + $lowA * $highB) + $lowA * $lowB;

        return [$product, $error];
    }

    /**
     * Add a * b to a compensated sum without losing the product's low bits.
     */
    public static function accumulate(CompensatedSum $sum, float $a, float $b): void
    {
        [$product, $error] = self::of($a, $b);

        $sum->add($product);
        $sum->add($error);
    }
}
