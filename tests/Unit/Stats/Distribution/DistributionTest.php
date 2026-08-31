<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit\Stats\Distribution;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Stats\Distribution\ChiSquared;
use Vegoia\Stats\Distribution\Distribution;
use Vegoia\Stats\Distribution\FisherSnedecor;
use Vegoia\Stats\Distribution\Normal;
use Vegoia\Stats\Distribution\StudentT;

/**
 * The contract, as distinct from the accuracy.
 *
 * Accuracy is settled in tests/Reference against mpmath and SciPy at 665
 * arguments. What is left is the shape of the thing: the parameters that make
 * no sense and must be refused, the boundaries where a formula built from
 * logarithms would return NAN, and the identities that hold whatever the
 * parameters are.
 */
#[CoversClass(Normal::class)]
#[CoversClass(StudentT::class)]
#[CoversClass(ChiSquared::class)]
#[CoversClass(FisherSnedecor::class)]
final class DistributionTest extends TestCase
{
    /** @return iterable<string, array{callable(): mixed}> */
    public static function impossibleParameters(): iterable
    {
        yield 'normal with zero spread' => [static fn () => new Normal(0.0, 0.0)];
        yield 'normal with negative spread' => [static fn () => new Normal(0.0, -1.0)];
        yield 'normal with infinite spread' => [static fn () => new Normal(0.0, INF)];
        yield 't with zero degrees of freedom' => [static fn () => new StudentT(0.0)];
        yield 't with negative degrees of freedom' => [static fn () => new StudentT(-3.0)];
        yield 'chi-squared with zero' => [static fn () => new ChiSquared(0.0)];
        yield 'F with zero numerator' => [static fn () => new FisherSnedecor(0.0, 5.0)];
        yield 'F with zero denominator' => [static fn () => new FisherSnedecor(5.0, 0.0)];
    }

    /** @param callable(): mixed $construct */
    #[DataProvider('impossibleParameters')]
    public function test_it_refuses_parameters_that_describe_no_distribution(callable $construct): void
    {
        $this->expectException(InvalidArgument::class);

        $construct();
    }

    /**
     * Each distribution with the infimum of its support, which is what the
     * zero quantile has to return: minus infinity for the two that live on
     * the whole line, zero for the two that do not.
     *
     * @return iterable<string, array{Distribution, float}>
     */
    public static function all(): iterable
    {
        yield 'normal' => [new Normal(), -INF];
        yield 'shifted normal' => [new Normal(3.0, 2.5), -INF];
        yield 't(1)' => [new StudentT(1.0), -INF];
        yield 't(30)' => [new StudentT(30.0), -INF];
        yield 'chi2(1)' => [new ChiSquared(1.0), 0.0];
        yield 'chi2(10)' => [new ChiSquared(10.0), 0.0];
        yield 'F(3, 10)' => [new FisherSnedecor(3.0, 10.0), 0.0];
        yield 'F(1, 1)' => [new FisherSnedecor(1.0, 1.0), 0.0];
    }

    #[DataProvider('all')]
    public function test_a_probability_outside_zero_and_one_is_refused(Distribution $distribution, float $infimum): void
    {
        $this->expectException(InvalidArgument::class);

        $distribution->upperQuantile(1.5);
    }

    #[DataProvider('all')]
    public function test_a_negative_probability_is_refused(Distribution $distribution, float $infimum): void
    {
        $this->expectException(InvalidArgument::class);

        $distribution->quantile(-0.1);
    }

    /**
     * The certain events, which no iteration should be entered for.
     */
    #[DataProvider('all')]
    public function test_the_two_certain_probabilities_are_answered_exactly(
        Distribution $distribution,
        float $infimum,
    ): void {
        self::assertSame(INF, $distribution->upperQuantile(0.0), 'nothing exceeds infinity');
        self::assertSame($infimum, $distribution->quantile(0.0), 'everything exceeds the infimum');
    }

    /**
     * The two tails must agree with each other and with the quantile.
     *
     * Deliberately at ordinary probabilities: the far tail is what the
     * reference tests are for, and this is here to catch a distribution whose
     * two halves were written against different conventions.
     */
    #[DataProvider('all')]
    public function test_the_quantile_inverts_both_tails(Distribution $distribution, float $infimum): void
    {
        foreach ([0.1, 0.25, 0.5, 0.75, 0.9] as $p) {
            $x = $distribution->quantile($p);

            self::assertEqualsWithDelta($p, $distribution->cumulative($x), 1.0e-11, "cdf(quantile({$p}))");
            self::assertEqualsWithDelta(
                1.0 - $p,
                $distribution->survival($x),
                1.0e-11,
                "sf(quantile({$p}))",
            );
        }
    }

    /** The cumulative never decreases, and the survival never increases. */
    #[DataProvider('all')]
    public function test_the_tails_are_monotone(Distribution $distribution, float $infimum): void
    {
        $previousCumulative = -INF;
        $previousSurvival = INF;

        foreach ([0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0] as $x) {
            $cumulative = $distribution->cumulative($x);
            $survival = $distribution->survival($x);

            self::assertGreaterThanOrEqual($previousCumulative, $cumulative, "cdf at {$x}");
            self::assertLessThanOrEqual($previousSurvival, $survival, "sf at {$x}");

            $previousCumulative = $cumulative;
            $previousSurvival = $survival;
        }
    }

    /** The density is never negative, whatever it is asked. */
    #[DataProvider('all')]
    public function test_the_density_is_never_negative(Distribution $distribution, float $infimum): void
    {
        foreach ([-100.0, -1.0, 0.0, 0.5, 1.0, 10.0, 1000.0] as $x) {
            self::assertGreaterThanOrEqual(0.0, $distribution->density($x), "density at {$x}");
        }
    }

    /**
     * Below its support the answer is known without any arithmetic, and the
     * general path would take a logarithm of a negative number to reach it.
     */
    public function test_the_positive_distributions_are_empty_below_zero(): void
    {
        foreach ([new ChiSquared(3.0), new FisherSnedecor(2.0, 7.0)] as $distribution) {
            self::assertSame(0.0, $distribution->cumulative(-1.0));
            self::assertSame(1.0, $distribution->survival(-1.0));
            self::assertSame(0.0, $distribution->density(-1.0));
            self::assertSame(0.0, $distribution->upperQuantile(1.0));
        }
    }

    /**
     * The chi-squared density at the origin has three answers depending on the
     * degrees of freedom, and a formula in logarithms gives none of them.
     */
    public function test_the_chi_squared_density_at_the_origin(): void
    {
        self::assertSame(INF, new ChiSquared(1.0)->density(0.0), 'unbounded below two');
        self::assertSame(0.5, new ChiSquared(2.0)->density(0.0), 'exactly a half at two');
        self::assertSame(0.0, new ChiSquared(3.0)->density(0.0), 'zero above two');
    }

    /** Student's t is symmetric about zero, exactly. */
    public function test_student_t_is_symmetric(): void
    {
        foreach ([1.0, 5.0, 100.0] as $df) {
            $t = new StudentT($df);

            self::assertSame(0.0, $t->upperQuantile(0.5), "median of t({$df})");

            foreach ([0.5, 2.0, 6.0] as $x) {
                self::assertSame($t->density($x), $t->density(-$x), "density at -{$x}");
                self::assertSame($t->cumulative(-$x), $t->survival($x), "cdf(-x) against sf(x)");
            }
        }
    }

    /**
     * With many degrees of freedom Student's t is the normal, and a
     * chi-squared over its degrees of freedom is too. Not an accuracy test --
     * it is the identity that says the parameter means what it should.
     */
    public function test_it_approaches_the_normal_as_the_degrees_of_freedom_grow(): void
    {
        $normal = new Normal();

        foreach ([-2.0, -0.5, 1.0, 2.5] as $x) {
            self::assertEqualsWithDelta(
                $normal->cumulative($x),
                new StudentT(1.0e6)->cumulative($x),
                1.0e-5,
                "t with a million degrees of freedom at {$x}",
            );
        }
    }

    /**
     * An F with one numerator degree of freedom is a squared t, which ties
     * two of these four together at every point rather than at a limit.
     */
    public function test_an_f_with_one_numerator_degree_of_freedom_is_a_squared_t(): void
    {
        $f = new FisherSnedecor(1.0, 12.0);
        $t = new StudentT(12.0);

        foreach ([0.5, 1.0, 2.0, 5.0] as $x) {
            self::assertEqualsWithDelta(
                2.0 * $t->survival(sqrt($x)),
                $f->survival($x),
                1.0e-14,
                "F(1, 12) survival at {$x}",
            );
        }
    }

    /** A chi-squared with two degrees of freedom is an exponential with mean two. */
    public function test_a_chi_squared_with_two_degrees_of_freedom_is_exponential(): void
    {
        $chi = new ChiSquared(2.0);

        foreach ([0.5, 1.0, 4.0, 20.0] as $x) {
            self::assertEqualsWithDelta(exp(-$x / 2.0), $chi->survival($x), 1.0e-15, "survival at {$x}");
        }
    }
}
