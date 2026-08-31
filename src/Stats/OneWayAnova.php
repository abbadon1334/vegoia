<?php

declare(strict_types=1);

namespace Vegoia\Stats;

use function array_filter;
use function array_values;
use function count;
use function sqrt;

use Vegoia\Exception\InvalidArgument;
use Vegoia\Stats\Distribution\FisherSnedecor;
use Vegoia\Support\CompensatedSum;

/**
 * One-way analysis of variance.
 *
 * Splits the total variation in a sample into the part explained by which
 * group an observation belongs to and the part left over, then asks whether
 * the first is large relative to the second. The F statistic is that ratio;
 * a large F means the group means differ by more than the within-group
 * scatter would account for.
 *
 * The sums of squares are accumulated about the means rather than from raw
 * sums of squares, for the reason the NIST ANOVA datasets exist to make: the
 * SmLs series holds values differing only in their last digits, and the
 * textbook computational formula loses every significant digit to
 * cancellation on them.
 *
 * @see https://www.itl.nist.gov/div898/strd/anova/anova.html
 */
final readonly class OneWayAnova
{
    private function __construct(
        public float $betweenSumOfSquares,
        public float $withinSumOfSquares,
        public int $betweenDegreesOfFreedom,
        public int $withinDegreesOfFreedom,
        public float $betweenMeanSquare,
        public float $withinMeanSquare,
        public float $fStatistic,
        public float $rSquared,
        public float $residualStandardDeviation,
        public int $groups,
        public int $observations,
    ) {
    }

    /**
     * @param list<list<float>> $groups one list of observations per group
     */
    public static function of(array $groups): self
    {
        $groups = array_values(array_filter($groups, static fn (array $g): bool => $g !== []));
        $groupCount = count($groups);

        if ($groupCount < 2) {
            throw InvalidArgument::tooFewValues('One-way ANOVA', $groupCount, 2);
        }

        $observations = 0;
        $grand = new CompensatedSum();

        foreach ($groups as $values) {
            $observations += count($values);

            foreach ($values as $value) {
                $grand->add($value);
            }
        }

        if ($observations <= $groupCount) {
            throw InvalidArgument::tooFewValues(
                'One-way ANOVA needs more observations than groups',
                $observations,
                $groupCount + 1,
            );
        }

        $grandMean = $grand->value() / $observations;

        // Recentre before accumulating. ANOVA is invariant under a shift of
        // every observation, so subtracting the grand mean changes nothing in
        // exact arithmetic -- and changes a great deal in floating point,
        // because the differences that carry all the signal are tiny beside
        // the values themselves. NIST's SmLs series is built from exactly that
        // situation: the same structure repeated at growing digit counts,
        // where the uncentred computation loses digit after digit.
        //
        // The residual grand mean of the shifted data is then near zero rather
        // than exactly zero; carrying it corrects for the error left in the
        // shift, the same trick the two-pass variance uses.
        $shifted = [];
        $residualGrand = new CompensatedSum();

        foreach ($groups as $index => $values) {
            $row = [];

            foreach ($values as $value) {
                $centred = $value - $grandMean;
                $row[] = $centred;
                $residualGrand->add($centred);
            }

            $shifted[$index] = $row;
        }

        $centredGrandMean = $residualGrand->value() / $observations;

        $between = new CompensatedSum();
        $within = new CompensatedSum();

        foreach ($shifted as $values) {
            $groupMean = Descriptive::of($values)->mean();
            $deviation = $groupMean - $centredGrandMean;

            $between->add(count($values) * $deviation * $deviation);

            foreach ($values as $value) {
                $residual = $value - $groupMean;
                $within->add($residual * $residual);
            }
        }

        $betweenSum = $between->value();
        $withinSum = $within->value();

        $betweenDf = $groupCount - 1;
        $withinDf = $observations - $groupCount;

        $betweenMean = $betweenSum / $betweenDf;
        $withinMean = $withinSum / $withinDf;

        $total = $betweenSum + $withinSum;

        return new self(
            $betweenSum,
            $withinSum,
            $betweenDf,
            $withinDf,
            $betweenMean,
            $withinMean,
            // A perfect fit leaves no within-group variation, and the ratio is
            // infinite rather than an error: the groups explain everything.
            $withinMean === 0.0 ? INF : $betweenMean / $withinMean,
            $total === 0.0 ? 0.0 : $betweenSum / $total,
            sqrt($withinMean),
            $groupCount,
            $observations,
        );
    }

    /**
     * Group observations by a parallel list of labels, then run the analysis.
     *
     * @param list<float>      $values
     * @param list<int|string> $labels
     */
    public static function grouped(array $values, array $labels): self
    {
        if (count($values) !== count($labels)) {
            throw InvalidArgument::malformedEdge(
                'ANOVA needs one label per observation: ' . count($values)
                . ' values and ' . count($labels) . ' labels'
            );
        }

        $groups = [];

        foreach ($values as $index => $value) {
            $groups[$labels[$index]][] = $value;
        }

        return self::of(array_values($groups));
    }

    /**
     * How often an F this large arises when every group has the same mean.
     *
     * The statistic on its own is not a result. An F of 21.0 sounds large and
     * an F of 1.18 sounds small, but neither is readable without the degrees
     * of freedom: the first is p = 2.6e-22 on (8, 180) and the second is
     * p = 0.35 on (4, 20), and only one of those is worth reporting.
     *
     * The upper tail, computed directly. A one-way analysis of variance is
     * one-sided by construction -- between-group variance can only inflate
     * the ratio, never deflate it -- and the interesting p-values live far
     * enough out that computing them as one minus the lower tail would return
     * zero for everything past about fifteen digits.
     */
    public function pValue(): float
    {
        return new FisherSnedecor(
            (float) $this->betweenDegreesOfFreedom,
            (float) $this->withinDegreesOfFreedom,
        )->survival($this->fStatistic);
    }
}
