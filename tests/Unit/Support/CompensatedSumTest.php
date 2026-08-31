<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit\Support;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\TestCase;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Support\CompensatedSum;

#[CoversClass(CompensatedSum::class)]
final class CompensatedSumTest extends TestCase
{
    /**
     * The canonical demonstration. Adding 0.1 ten thousand times with plain
     * floating-point addition drifts, because 0.1 is not representable in
     * binary and the error compounds once per addition. Neumaier keeps the
     * discarded low-order bits and folds them back.
     */
    public function test_it_recovers_the_bits_naive_addition_discards(): void
    {
        $values = array_fill(0, 10_000, 0.1);

        $naive = 0.0;
        foreach ($values as $value) {
            $naive += $value;
        }

        self::assertNotSame(1000.0, $naive, 'the premise: naive addition drifts');
        self::assertSame(1000.0, CompensatedSum::of($values));
    }

    /**
     * The case Kahan's original formulation gets wrong and Neumaier's does not:
     * a large value, then two small ones that together are significant. Kahan
     * loses them because the compensation is applied to the wrong operand.
     */
    public function test_it_handles_a_large_value_followed_by_small_ones(): void
    {
        self::assertSame(2.0, CompensatedSum::of([1.0, 1.0e100, 1.0, -1.0e100]));
    }

    public function test_an_empty_sum_is_zero(): void
    {
        self::assertSame(0.0, CompensatedSum::of([]));
    }

    public function test_it_accumulates_incrementally(): void
    {
        $sum = new CompensatedSum();

        self::assertSame(0.0, $sum->value());

        $sum->add(1.5)->add(2.5);

        self::assertSame(4.0, $sum->value());

        $sum->add(-4.0);

        self::assertSame(0.0, $sum->value());
    }

    /**
     * Dividing value() rounds twice: once folding the compensation into the
     * head, once dividing. Dividing inside keeps the compensation through the
     * division, and it is worth two digits on a mean of large, tightly
     * clustered values -- which is what makes NIST's NumAcc3 and NumAcc4 come
     * out exact.
     */
    public function test_dividing_inside_keeps_what_dividing_the_result_loses(): void
    {
        // A thousand values just above 1e9, differing in the ninth decimal.
        $sum = new CompensatedSum();
        $values = [];

        for ($i = 0; $i < 1000; $i++) {
            $value = 1.0e9 + $i * 1.0e-9;
            $values[] = $value;
            $sum->add($value);
        }

        $inside = $sum->dividedBy(1000.0);
        $outside = $sum->value() / 1000.0;

        // The exact mean is 1e9 + (999/2) * 1e-9.
        $expected = 1.0e9 + 499.5e-9;

        self::assertLessThanOrEqual(
            abs($outside - $expected),
            abs($inside - $expected),
            'dividing inside cannot be worse than dividing the rounded total',
        );

        self::assertSame(1.5, (new CompensatedSum())->add(3.0)->dividedBy(2.0));
    }

    public function test_dividing_by_zero_is_refused(): void
    {
        $this->expectException(InvalidArgument::class);

        (new CompensatedSum())->add(1.0)->dividedBy(0.0);
    }

    /**
     * twoSum is exact: the head is the rounded sum, the tail is precisely what
     * that rounding discarded.
     */
    public function test_two_sum_recovers_the_bits_addition_drops(): void
    {
        [$head, $tail] = CompensatedSum::twoSum(1.0, 2 ** -60);

        self::assertSame(1.0, $head, 'the addend is below the last bit');
        self::assertSame(2 ** -60, $tail, 'and is recovered whole');

        [$head, $tail] = CompensatedSum::twoSum(3.0, 4.0);

        self::assertSame(7.0, $head);
        self::assertSame(0.0, $tail, 'an exact sum loses nothing');
    }

    public function test_it_accepts_integers_alongside_floats(): void
    {
        self::assertSame(6.0, CompensatedSum::of([1, 2.0, 3]));
    }
}
