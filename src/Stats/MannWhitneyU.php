<?php

declare(strict_types=1);

namespace Vegoia\Stats;

use function abs;
use function array_merge;
use function array_slice;
use function count;
use function min;
use function sqrt;

use Vegoia\Exception\InvalidArgument;
use Vegoia\Stats\Distribution\Normal;
use Vegoia\Support\CompensatedSum;

/**
 * Does one sample tend to hold larger values than the other?
 *
 * The rank test to reach for when a t-test's assumptions are not available:
 * it asks nothing about the shape of the distributions, only about the order
 * of the observations, so a single wild value moves it by one rank rather than
 * by however large it was.
 *
 * The normal approximation only. SciPy offers an exact route as well and
 * switches between them on sample size and on whether ties are present, and
 * the two disagree materially -- on one sample here, 0.114 against 0.151. That
 * makes `auto` a property of SciPy's heuristic rather than of the mathematics,
 * so the fixtures for this class are generated with `method='asymptotic'`
 * passed explicitly and this implements that and says so. The exact route is a
 * different procedure, not a refinement of this one, and would arrive beside
 * it rather than inside it.
 */
final readonly class MannWhitneyU
{
    private function __construct(
        public float $statistic,
        public float $z,
        public float $mean,
        public float $variance,
        public float $tieCorrection,
        public int $firstCount,
        public int $secondCount,
        public Alternative $alternative,
        public Continuity $continuity,
    ) {
    }

    /**
     * @param list<float> $x
     * @param list<float> $y
     */
    public static function of(
        array $x,
        array $y,
        Alternative $alternative = Alternative::TwoSided,
        Continuity $continuity = Continuity::Corrected,
    ): self {
        $nx = count($x);
        $ny = count($y);

        if ($nx < 1) {
            throw InvalidArgument::tooFewValues('Mann-Whitney needs a first sample that', $nx, 1);
        }

        if ($ny < 1) {
            throw InvalidArgument::tooFewValues('Mann-Whitney needs a second sample that', $ny, 1);
        }

        $pooled = array_merge($x, $y);
        $n = $nx + $ny;

        $ranks = Ranks::midranks($pooled);

        $rankSum = new CompensatedSum();

        foreach (array_slice($ranks, 0, $nx) as $rank) {
            $rankSum->add($rank);
        }

        // U for the FIRST sample, which is what scipy.stats.mannwhitneyu(x, y)
        // reports. U for the second is n1*n2 - U1 and is one line away.
        $statistic = $rankSum->value() - $nx * ($nx + 1) / 2.0;
        $mean = $nx * $ny / 2.0;

        // The tie correction always applies. With nothing tied the sum is zero
        // and this reduces to the textbook n1*n2*(N+1)/12, so there is one
        // formula and no special case to get wrong.
        $tied = 0;

        foreach (Ranks::tieSizes($pooled) as $size) {
            $tied += $size * $size * $size - $size;
        }

        // n is at least 2: both samples were required to be non-empty.
        $tieCorrection = $tied / ($n * ($n - 1));
        $variance = $nx * $ny / 12.0 * (($n + 1) - $tieCorrection);

        if ($variance <= 0.0) {
            throw InvalidArgument::malformedEdge(
                'Mann-Whitney is undefined when every observation is tied: there is no ordering '
                . 'information at all, so the statistic is not extreme, it is meaningless'
            );
        }

        $deviation = match ($alternative) {
            Alternative::TwoSided => abs($statistic - $mean),
            Alternative::Greater => $statistic - $mean,
            Alternative::Less => $mean - $statistic,
        };

        // The half-step moves the statistic towards the null, whichever
        // direction the question faces.
        if ($continuity === Continuity::Corrected) {
            $deviation -= 0.5;
        }

        return new self(
            $statistic,
            $deviation / sqrt($variance),
            $mean,
            $variance,
            $tieCorrection,
            $nx,
            $ny,
            $alternative,
            $continuity,
        );
    }

    public function pValue(): float
    {
        $tail = new Normal()->survival($this->z);

        return min(1.0, $this->alternative === Alternative::TwoSided ? 2.0 * $tail : $tail);
    }
}
