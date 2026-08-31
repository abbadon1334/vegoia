<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit\Stats;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Stats\SpecialFunction;

/**
 * The parts of the contract that are not about accuracy.
 *
 * Accuracy is settled next door in tests/Reference against mpmath and SciPy.
 * What is left is the boundary: the arguments where these functions are
 * undefined and must say so rather than return a quiet NAN, and the handful
 * of points where the answer is known exactly and no approximation should be
 * consulted at all.
 */
#[CoversClass(SpecialFunction::class)]
final class SpecialFunctionTest extends TestCase
{
    /** @return iterable<string, array{callable(): mixed, string}> */
    public static function undefinedArguments(): iterable
    {
        yield 'logGamma at zero' => [static fn () => SpecialFunction::logGamma(0.0), 'positive'];
        yield 'logGamma below zero' => [static fn () => SpecialFunction::logGamma(-1.5), 'positive'];
        yield 'gammaP with a = 0' => [static fn () => SpecialFunction::regularizedGammaP(0.0, 1.0), 'positive'];
        yield 'gammaP with negative x' => [static fn () => SpecialFunction::regularizedGammaP(1.0, -1.0), 'negative'];
        yield 'gammaQ with a = 0' => [static fn () => SpecialFunction::regularizedGammaQ(0.0, 1.0), 'positive'];
        yield 'gammaQ with negative x' => [static fn () => SpecialFunction::regularizedGammaQ(1.0, -1.0), 'negative'];
        yield 'beta with a = 0' => [static fn () => SpecialFunction::regularizedBeta(0.5, 0.0, 1.0), 'positive'];
        yield 'beta with b = 0' => [static fn () => SpecialFunction::regularizedBeta(0.5, 1.0, 0.0), 'positive'];
        yield 'beta below zero' => [static fn () => SpecialFunction::regularizedBeta(-0.1, 1.0, 1.0), '[0, 1]'];
        yield 'beta above one' => [static fn () => SpecialFunction::regularizedBeta(1.1, 1.0, 1.0), '[0, 1]'];
    }

    /** @param callable(): mixed $call */
    #[DataProvider('undefinedArguments')]
    public function test_it_refuses_arguments_it_is_not_defined_for(callable $call, string $because): void
    {
        $this->expectException(InvalidArgument::class);
        $this->expectExceptionMessageMatches('/' . preg_quote($because, '/') . '/');

        $call();
    }

    /**
     * The endpoints of the incomplete gamma, where no series is needed.
     *
     * P(a, 0) is 0 and Q(a, 0) is 1 for every a, and getting these from the
     * general path would mean evaluating log(0).
     */
    public function test_the_incomplete_gamma_is_exact_at_zero(): void
    {
        foreach ([0.5, 1.0, 7.0, 1000.0] as $a) {
            self::assertSame(0.0, SpecialFunction::regularizedGammaP($a, 0.0), "P({$a}, 0)");
            self::assertSame(1.0, SpecialFunction::regularizedGammaQ($a, 0.0), "Q({$a}, 0)");
        }
    }

    /** I_0(a, b) is 0 and I_1(a, b) is 1, whatever a and b are. */
    public function test_the_incomplete_beta_is_exact_at_both_endpoints(): void
    {
        foreach ([[0.5, 0.5], [1.0, 1.0], [30.0, 2.0]] as [$a, $b]) {
            self::assertSame(0.0, SpecialFunction::regularizedBeta(0.0, $a, $b), "I_0({$a}, {$b})");
            self::assertSame(1.0, SpecialFunction::regularizedBeta(1.0, $a, $b), "I_1({$a}, {$b})");
        }
    }

    /** erf is odd and erfc(x) + erfc(-x) is 2, which pins the negative branch. */
    public function test_the_error_function_is_odd(): void
    {
        foreach ([0.25, 1.0, 2.5, 4.0] as $x) {
            self::assertEqualsWithDelta(
                -SpecialFunction::erf($x),
                SpecialFunction::erf(-$x),
                1.0e-15,
                "erf(-{$x})",
            );

            self::assertEqualsWithDelta(
                2.0,
                SpecialFunction::erfc($x) + SpecialFunction::erfc(-$x),
                1.0e-15,
                "erfc({$x}) + erfc(-{$x})",
            );
        }
    }

    public function test_erf_is_zero_at_zero(): void
    {
        self::assertSame(0.0, SpecialFunction::erf(0.0));
        self::assertSame(1.0, SpecialFunction::erfc(0.0));
    }

    /**
     * The far tail must not collapse to zero.
     *
     * erfc(20) is about 5.4e-176, which a double holds comfortably; an
     * implementation that computes it as 1 - erf(20) returns exactly zero,
     * and every p-value past six sigma with it.
     */
    public function test_the_far_tail_survives(): void
    {
        self::assertGreaterThan(0.0, SpecialFunction::erfc(20.0));
        self::assertGreaterThan(0.0, SpecialFunction::erfc(26.0));
        self::assertGreaterThan(0.0, SpecialFunction::regularizedGammaQ(1.0, 500.0));
    }
}
