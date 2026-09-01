<?php

declare(strict_types=1);

namespace Vegoia\Stats;

/**
 * Whether a discrete statistic is read as though it were continuous.
 *
 * A statistic that can only land on whole or half steps is being compared
 * against a distribution that fills the gaps between them, and the correction
 * moves the observed value half a step towards the null before asking. It
 * matters most where it is least welcome: on small samples, where the gaps are
 * widest and the uncorrected p-value is the optimistic one.
 *
 * Corrected is the default throughout, which is SciPy's choice and R's.
 */
enum Continuity
{
    /** Yates' correction on a 2x2 table; the half-step on Mann-Whitney's U. */
    case Corrected;

    case Uncorrected;
}
