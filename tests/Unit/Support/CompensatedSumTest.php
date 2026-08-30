<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit\Support;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\TestCase;
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

    public function test_it_accepts_integers_alongside_floats(): void
    {
        self::assertSame(6.0, CompensatedSum::of([1, 2.0, 3]));
    }
}
