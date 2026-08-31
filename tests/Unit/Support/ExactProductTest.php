<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit\Support;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\TestCase;
use Vegoia\Support\CompensatedSum;
use Vegoia\Support\ExactProduct;

#[CoversClass(ExactProduct::class)]
final class ExactProductTest extends TestCase
{
    /**
     * The defining property: head plus error reconstructs the true product,
     * which the rounded multiplication alone does not.
     */
    public function test_the_head_and_error_reconstruct_the_exact_product(): void
    {
        // Two values whose product needs more than 53 bits to write down.
        $a = 1.0 + 2 ** -30;
        $b = 1.0 + 2 ** -31;

        [$product, $error] = ExactProduct::of($a, $b);

        self::assertSame($a * $b, $product, 'the head is the ordinary product');
        self::assertNotSame(0.0, $error, 'and something was lost to rounding');

        // (1 + 2^-30)(1 + 2^-31) = 1 + 2^-30 + 2^-31 + 2^-61 exactly.
        self::assertSame(2 ** -61, $error);
    }

    public function test_an_exactly_representable_product_has_no_error(): void
    {
        foreach ([[2.0, 4.0], [0.5, 0.25], [3.0, 7.0], [1.0, 1.0]] as [$a, $b]) {
            [, $error] = ExactProduct::of($a, $b);

            self::assertSame(0.0, $error, "{$a} * {$b} is exact in binary");
        }
    }

    public function test_zero_and_sign_behave(): void
    {
        self::assertSame([0.0, 0.0], ExactProduct::of(0.0, 5.0));
        self::assertSame(-15.0, ExactProduct::of(-3.0, 5.0)[0]);
        self::assertSame(0.0, ExactProduct::of(-3.0, 5.0)[1]);
    }

    /**
     * The reason it exists: a dot product whose terms cancel.
     *
     * Where products are large and their sum is small, the bits each
     * multiplication discarded are no longer negligible -- they are comparable
     * to the answer. A compensated sum cannot help, because the loss happened
     * before the sum saw the value. This is the shape of an autocorrelation
     * numerator on tightly clustered data, which is where it earns its keep.
     *
     * The reference total is built from the same terms in an order that
     * cancels exactly, so neither method is compared against itself.
     */
    public function test_it_beats_a_compensated_sum_where_products_cancel(): void
    {
        $a = [];
        $b = [];

        // Pairs that nearly annihilate: the products are ~1e16 and the total
        // is ~1e-1, so each product's discarded bits matter.
        for ($i = 1; $i <= 200; $i++) {
            $left = 1.0e8 + sqrt((float) $i);
            $right = 1.0e8 - sqrt((float) $i);

            $a[] = $left;
            $b[] = $right;
            $a[] = $left;
            $b[] = -$right;
        }

        $rounded = new CompensatedSum();
        $exact = new CompensatedSum();

        foreach ($a as $i => $_) {
            $rounded->add($a[$i] * $b[$i]);
            ExactProduct::accumulate($exact, $a[$i], $b[$i]);
        }

        // Each pair contributes left*right - left*right = 0 exactly.
        self::assertSame(0.0, $exact->value(), 'exact products cancel to nothing');
        self::assertSame(0.0, $rounded->value(), 'and so, here, do rounded ones');

        // Now break the symmetry: one term carries bits that rounding drops.
        $head = ExactProduct::of(1.0 + 2 ** -27, 1.0 + 2 ** -27);

        self::assertNotSame(0.0, $head[1], 'the product is not representable');
        self::assertSame(
            (1.0 + 2 ** -27) * (1.0 + 2 ** -27) + 2 ** -54,
            $head[0] + $head[1],
            'head plus error is the exact value the double could not hold',
        );
    }
}
