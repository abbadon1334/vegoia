<?php

declare(strict_types=1);

namespace Vegoia\Stats;

use function abs;
use function exp;
use function log;
use function log1p;

use Vegoia\Exception\InvalidArgument;
use Vegoia\Support\CompensatedSum;
use Vegoia\Support\ExactProduct;

/**
 * The special functions every statistical distribution is built on.
 *
 * PHP provides none of them. There is no erf, no erfc, no lgamma, no
 * incomplete gamma or beta in the language or in any extension one can
 * reasonably expect to find installed, which is the concrete reason a PHP
 * program cannot turn an F statistic into a p-value without either shelling
 * out or writing this file.
 *
 * Two things matter for accuracy, and both cost more code than the textbook
 * formula:
 *
 * The first is choosing a representation per region. The incomplete gamma has
 * a series that converges quickly below x = a + 1 and a continued fraction
 * that converges quickly above it, and each is nearly useless in the other's
 * territory. The same is true of the incomplete beta either side of
 * (a+1)/(a+b+2).
 *
 * The second is never computing a small number as one minus a large one.
 * Q(a, x) is not 1 - P(a, x): at x = 2000, a = 1000 the true Q is around
 * 1e-108, and the subtraction returns zero having thrown away all 108 digits.
 * The upper and lower tails are therefore computed by different routes and
 * only converted into one another where the value being converted is the
 * larger of the two.
 *
 * Accuracy is measured, not asserted: tests/Reference/Stats compares every
 * function against mpmath at 50 digits and requires it to come within half a
 * digit of SciPy at 224 arguments.
 */
final class SpecialFunction
{
    /** Iterations before the series or the fraction is declared not to converge. */
    private const int MAX_ITERATIONS = 300;

    /** Relative size at which another term cannot change the double. */
    private const float EPSILON = 1.0e-16;

    /** Smallest positive normal double, used to keep Lentz's method off zero. */
    private const float TINY = 1.0e-300;

    /** log(sqrt(2 * pi)), the constant in front of the Lanczos series. */
    private const float LOG_SQRT_TWO_PI = 0.91893853320467274178;

    /**
     * Lanczos approximation, g = 671/128 with fifteen terms.
     *
     * The parameters are Numerical Recipes' third edition set, chosen there
     * because they hold about fifteen digits across the positive reals -- one
     * more than the classic g = 5 set, which runs out at twelve and would put
     * the incomplete gamma below SciPy on the large-parameter arguments.
     */
    private const array LANCZOS = [
        57.1562356658629235,
        -59.5979603554754912,
        14.1360979747417471,
        -0.491913816097620199,
        0.339946499848118887e-4,
        0.465236289270485756e-4,
        -0.983744753048795646e-4,
        0.158088703224912494e-3,
        -0.210264441724104883e-3,
        0.217439618115212643e-3,
        -0.164318106536763890e-3,
        0.844182239838527433e-4,
        -0.261908384015814087e-4,
        0.368991826595316234e-5,
    ];

    /** The Lanczos parameter g, 671/128. */
    private const float G = 5.2421875;

    /** The natural logarithm of the gamma function, for x > 0. */
    public static function logGamma(float $x): float
    {
        if ($x <= 0.0) {
            throw InvalidArgument::outOfDomain('logGamma', $x, 'x must be positive');
        }

        $shifted = $x + self::G;

        return ($x + 0.5) * log($shifted) - $shifted
            + self::LOG_SQRT_TWO_PI + log(self::lanczosSeries($x) / $x);
    }

    /**
     * The Lanczos series S, defined so that
     * gamma(x) = (x+g)^(x+1/2) e^-(x+g) sqrt(2pi) S(x) / x.
     *
     * Exposed separately from logGamma because the prefactors below need S
     * itself rather than a logarithm that has already been added to something
     * large.
     */
    private static function lanczosSeries(float $x): float
    {
        $series = 0.999999999999997092;
        $denominator = $x;

        foreach (self::LANCZOS as $coefficient) {
            $series += $coefficient / ++$denominator;
        }

        return $series;
    }

    /**
     * log(x^a e^-x / gamma(a)), the factor in front of both incomplete gammas.
     *
     * Written this way rather than as `a*log(x) - x - logGamma(a)` because
     * that expression is a subtraction of nearly equal large numbers. At
     * a = 1000, x = 1000 its terms are 6907.75, -1000 and -5905.22 and the
     * answer is 2.53: four of the sixteen digits survive, and the incomplete
     * gamma inherits the loss. Measured, that cost 3 digits against SciPy and
     * failed the reference test.
     *
     * Substituting the Lanczos form of gamma(a) and cancelling by hand gives
     * an expression whose largest intermediate term is around 7 instead of
     * 6900, so nothing large is ever subtracted from anything large.
     */
    private static function logGammaPrefix(float $a, float $x): CompensatedSum
    {
        $shifted = $a + self::G;

        [$logarithm, $correction] = self::logRatio($x, 1.0, $shifted);

        $total = new CompensatedSum();
        ExactProduct::accumulate($total, $a, $logarithm);
        ExactProduct::accumulate($total, $a, $correction);

        return $total
            ->add($shifted - $x)
            ->add(-0.5 * log($shifted))
            ->add(-self::LOG_SQRT_TWO_PI)
            ->add(-log(self::lanczosSeries($a) / $a));
    }

    /**
     * log(x^a (1-x)^b / B(a, b)), the factor in front of the incomplete beta.
     *
     * The same cancellation, and the same cure. At a = b = 100, x = 0.5 the
     * naive form subtracts 139.66 from 138.63 to reach 1.03, which measured
     * 12.86 digits against SciPy's 15.11. In ratio form the largest term is
     * about 5.
     *
     * `$complement` exists because 1 - x is its own loss when x is near one,
     * and a caller often knows the difference exactly while the subtraction
     * cannot. The F distribution's argument at (100, 10) degrees of freedom
     * and x = 100 is z = 10000/10010; forming 1 - z leaves 13.47 digits,
     * which the b = 5 exponent turns into a density accurate to 12.78 against
     * SciPy's 13.33. Passing d2/(d1 x + d2) instead costs nothing and loses
     * nothing.
     */
    private static function logBetaPrefix(
        float $x,
        float $a,
        float $b,
        ?float $complement = null,
    ): CompensatedSum {
        $oneMinusX = $complement ?? 1.0 - $x;
        $c = $a + $b;
        $shiftedA = $a + self::G;
        $shiftedB = $b + self::G;
        $shiftedC = $c + self::G;

        [$logA, $correctionA] = self::logRatio($x, $shiftedC, $shiftedA);
        [$logB, $correctionB] = self::logRatio($oneMinusX, $shiftedC, $shiftedB);

        $total = new CompensatedSum();
        ExactProduct::accumulate($total, $a, $logA);
        ExactProduct::accumulate($total, $a, $correctionA);
        ExactProduct::accumulate($total, $b, $logB);
        ExactProduct::accumulate($total, $b, $correctionB);

        return $total
            ->add(0.5 * log($shiftedC / ($shiftedA * $shiftedB)))
            ->add(self::G)
            ->add(-self::LOG_SQRT_TWO_PI)
            ->add(log(
                self::lanczosSeries($c) * $a * $b
                / (self::lanczosSeries($a) * self::lanczosSeries($b) * $c)
            ));
    }

    /**
     * The error function.
     *
     * Expressed through the incomplete gamma rather than its own series,
     * because erf(x) is P(1/2, x^2) exactly and the region switching is then
     * inherited instead of duplicated.
     */
    public static function erf(float $x): float
    {
        if ($x === 0.0) {
            return 0.0;
        }

        return $x > 0.0
            ? self::regularizedGammaP(0.5, $x * $x)
            : -self::regularizedGammaP(0.5, $x * $x);
    }

    /**
     * The complementary error function, 1 - erf(x) without the cancellation.
     *
     * At x = 20 the true value is near 5e-176. Computing it as 1 - erf(20)
     * gives exactly zero, since erf(20) is 1 to every bit a double has. This
     * is the routine a normal-tail p-value has to go through, so it is the
     * one that has to be right.
     */
    public static function erfc(float $x): float
    {
        if ($x >= 0.0) {
            return self::regularizedGammaQ(0.5, $x * $x);
        }

        return 1.0 + self::regularizedGammaP(0.5, $x * $x);
    }

    /**
     * x^a e^-x / gamma(a), the density of the gamma distribution without its
     * scale, and the factor in front of both incomplete gammas.
     *
     * Public because the chi-squared density is exactly this over x, and
     * writing that density out again from logarithms would repeat the
     * cancellation this class exists to avoid -- at 500 degrees of freedom
     * logGamma(250) is 1128, and forming the density from it directly costs
     * the last three digits.
     */
    public static function gammaPrefactor(float $a, float $x): float
    {
        self::assertGammaArguments($a, $x);

        if ($x === 0.0) {
            return 0.0;
        }

        return self::logGammaPrefix($a, $x)->exponentiated();
    }

    /**
     * x^a (1-x)^b / B(a, b), the factor in front of the incomplete beta.
     *
     * Public for the same reason: the F density is this over x, and the
     * Student's t density is this over |t|. Both were written out from
     * logGamma first, and both measured about a digit short of SciPy until
     * they were routed through here.
     */
    public static function betaPrefactor(float $x, float $a, float $b, ?float $complement = null): float
    {
        if ($a <= 0.0 || $b <= 0.0) {
            throw InvalidArgument::outOfDomain('betaPrefactor', $a, 'a and b must be positive');
        }

        if ($x <= 0.0 || $x >= 1.0) {
            return 0.0;
        }

        return self::logBetaPrefix($x, $a, $b, $complement)->exponentiated();
    }

    /** The regularized lower incomplete gamma, P(a, x). */
    public static function regularizedGammaP(float $a, float $x): float
    {
        self::assertGammaArguments($a, $x);

        if ($x === 0.0) {
            return 0.0;
        }

        // Below a + 1 the series converges in a few terms; above it, the
        // fraction does, and the lower tail is then the larger of the two so
        // the subtraction is safe.
        return $x < $a + 1.0
            ? self::lowerSeries($a, $x)
            : 1.0 - self::upperFraction($a, $x);
    }

    /** The regularized upper incomplete gamma, Q(a, x) = 1 - P(a, x). */
    public static function regularizedGammaQ(float $a, float $x): float
    {
        self::assertGammaArguments($a, $x);

        if ($x === 0.0) {
            return 1.0;
        }

        return $x < $a + 1.0
            ? 1.0 - self::lowerSeries($a, $x)
            : self::upperFraction($a, $x);
    }

    /**
     * The regularized incomplete beta, I_x(a, b).
     *
     * This is the function behind Student's t, the F distribution and the
     * binomial tail; everything else in this class exists to support it or to
     * stand beside it.
     */
    /**
     * @param float|null $complement 1 - x, when the caller can form it more
     *                               accurately than the subtraction can. See
     *                               logBetaPrefix for why that is worth an
     *                               argument.
     */
    public static function regularizedBeta(float $x, float $a, float $b, ?float $complement = null): float
    {
        if ($a <= 0.0 || $b <= 0.0) {
            throw InvalidArgument::outOfDomain('regularizedBeta', $a, 'a and b must be positive');
        }

        if ($x < 0.0 || $x > 1.0) {
            throw InvalidArgument::outOfDomain('regularizedBeta', $x, 'x must lie in [0, 1]');
        }

        if ($x === 0.0 || $x === 1.0) {
            return $x;
        }

        $oneMinusX = $complement ?? 1.0 - $x;
        $total = self::logBetaPrefix($x, $a, $b, $complement);

        // Underflow here is the honest answer, not a failure: the true value
        // has no double. exp() would return 0.0 anyway, but the continued
        // fraction below is not worth entering. Which end it collapses to
        // follows the same split as the branch below.
        if ($total->value() < -745.0) {
            return $x < $a / ($a + $b) ? 0.0 : 1.0;
        }

        $prefactor = $total->exponentiated();

        // Which side to compute directly, and which to reach by the
        // reflection I_x(a,b) = 1 - I_{1-x}(b,a).
        //
        // The textbook criterion is the mode, (a+1)/(a+b+2), and it is chosen
        // for how fast the continued fraction converges rather than for how
        // much the answer is worth once it arrives. Those differ. At
        // a = 500, b = 0.5, x = 0.9973 the mode sends this to the reflection,
        // where I_{1-x}(b,a) is 0.9 and the subtraction from one turns its
        // relative error into nine times as much -- Student's t at 1000
        // degrees of freedom then reported its 5% point to 12.93 digits
        // against SciPy's 15.95.
        //
        // The mean, a/(a+b), is 0.999 there and keeps the direct branch,
        // which computes the 0.1 as itself. It costs a few more iterations of
        // a fraction that converges either way, and it is what Boost switches
        // on for the same reason. Measured across all 665 reference points,
        // it loses nothing anywhere else.
        return $x < $a / ($a + $b)
            ? $prefactor * self::betaFraction($x, $a, $b) / $a
            : 1.0 - $prefactor * self::betaFraction($oneMinusX, $b, $a) / $b;
    }

    /**
     * log(factor * scale / divisor), with the arithmetic's own rounding
     * recovered rather than absorbed.
     *
     * Two things are going on, and both were forced by measurement.
     *
     * The first is the choice between log1p and log. log1p on the deviation
     * from 1 is the accurate one when the ratio is near 1, which is where the
     * plain logarithm loses the leading digits to the subtraction hidden
     * inside it. Away from 1 they swap places and badly: at x = 1e-16 against
     * a shift of 5.74 the deviation rounds to -1, log1p is handed a zero and
     * every digit is gone -- taking log1p everywhere cost erf(1e-8) all
     * sixteen of them.
     *
     * The second is the correction. Callers multiply this logarithm by a
     * shape parameter, and that parameter can be 500. The quotient carries a
     * relative error of about 1e-16 from its own two roundings, the logarithm
     * inherits it as an absolute error, and multiplying by 500 turns it into
     * 1e-13 -- which is what capped the tail of Student's t at 1000 degrees of
     * freedom to 12.99 digits against SciPy's 15.95. The product and the
     * division are therefore both done in a form that hands back what they
     * discarded, and the caller adds `parameter * correction` alongside.
     *
     * @return array{float, float} the logarithm, and a relative correction to
     *                             be added to it before scaling
     */
    private static function logRatio(float $factor, float $scale, float $divisor): array
    {
        [$product, $productError] = ExactProduct::of($factor, $scale);
        $quotient = $product / $divisor;
        [$back, $backError] = ExactProduct::of($quotient, $divisor);

        // product - quotient * divisor, exactly: what the division threw away.
        // The subtraction itself is exact, the two being within a factor of
        // two of each other by construction.
        $correction = ((($product - $back) - $backError) + $productError) / $product;

        $deviation = $quotient - 1.0;

        return [abs($deviation) < 0.5 ? log1p($deviation) : log($quotient), $correction];
    }

    /**
     * P(a, x) by its power series, for x below a + 1.
     *
     * The terms are all positive, so nothing cancels and the sum is as
     * accurate as the prefactor in front of it.
     */
    private static function lowerSeries(float $a, float $x): float
    {
        $total = self::logGammaPrefix($a, $x);

        if ($total->value() < -745.0) {
            return 0.0;
        }

        $denominator = $a;
        $term = 1.0 / $a;
        $sum = $term;

        for ($i = 0; $i < self::MAX_ITERATIONS; $i++) {
            $term *= $x / ++$denominator;
            $sum += $term;

            if (abs($term) < abs($sum) * self::EPSILON) {
                break;
            }
        }

        return $sum * $total->exponentiated();
    }

    /**
     * Q(a, x) by its continued fraction, for x at or above a + 1.
     *
     * Evaluated by the modified Lentz method, which builds the fraction from
     * the front and so does not need to be told in advance how deep to go.
     * The guard against a zero denominator is what "modified" refers to.
     */
    private static function upperFraction(float $a, float $x): float
    {
        $total = self::logGammaPrefix($a, $x);

        if ($total->value() < -745.0) {
            return 0.0;
        }

        $b = $x + 1.0 - $a;
        $c = 1.0 / self::TINY;
        $d = 1.0 / $b;
        $fraction = $d;

        for ($i = 1; $i <= self::MAX_ITERATIONS; $i++) {
            $an = -$i * ($i - $a);
            $b += 2.0;

            $d = $an * $d + $b;
            if (abs($d) < self::TINY) {
                $d = self::TINY;
            }

            $c = $b + $an / $c;
            if (abs($c) < self::TINY) {
                $c = self::TINY;
            }

            $d = 1.0 / $d;
            $delta = $d * $c;
            $fraction *= $delta;

            if (abs($delta - 1.0) < self::EPSILON) {
                break;
            }
        }

        return $fraction * $total->exponentiated();
    }

    /**
     * The continued fraction for the incomplete beta, again by modified Lentz.
     *
     * The even and odd terms have different forms, which is why the loop
     * advances two steps per iteration rather than one.
     */
    private static function betaFraction(float $x, float $a, float $b): float
    {
        $qab = $a + $b;
        $qap = $a + 1.0;
        $qam = $a - 1.0;

        $c = 1.0;
        $d = 1.0 - $qab * $x / $qap;

        if (abs($d) < self::TINY) {
            $d = self::TINY;
        }

        $d = 1.0 / $d;
        $fraction = $d;

        for ($m = 1; $m <= self::MAX_ITERATIONS; $m++) {
            $m2 = 2 * $m;

            // Even step.
            $aa = $m * ($b - $m) * $x / (($qam + $m2) * ($a + $m2));
            $d = 1.0 + $aa * $d;
            if (abs($d) < self::TINY) {
                $d = self::TINY;
            }
            $c = 1.0 + $aa / $c;
            if (abs($c) < self::TINY) {
                $c = self::TINY;
            }
            $d = 1.0 / $d;
            $fraction *= $d * $c;

            // Odd step.
            $aa = -($a + $m) * ($qab + $m) * $x / (($a + $m2) * ($qap + $m2));
            $d = 1.0 + $aa * $d;
            if (abs($d) < self::TINY) {
                $d = self::TINY;
            }
            $c = 1.0 + $aa / $c;
            if (abs($c) < self::TINY) {
                $c = self::TINY;
            }
            $d = 1.0 / $d;
            $delta = $d * $c;
            $fraction *= $delta;

            if (abs($delta - 1.0) < self::EPSILON) {
                break;
            }
        }

        return $fraction;
    }

    private static function assertGammaArguments(float $a, float $x): void
    {
        if ($a <= 0.0) {
            throw InvalidArgument::outOfDomain('incomplete gamma', $a, 'a must be positive');
        }

        if ($x < 0.0) {
            throw InvalidArgument::outOfDomain('incomplete gamma', $x, 'x must not be negative');
        }
    }
}
