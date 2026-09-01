<?php

declare(strict_types=1);

namespace Vegoia\Stats;

/**
 * Which direction a test is being asked about.
 *
 * Less and Greater name what the *first* sample does: Less asks whether the
 * values behind x tend to fall below those behind y. That is SciPy's
 * convention, and it is the one that reads correctly when the samples are
 * named rather than numbered.
 *
 * TwoSided is the default. A one-sided test is twice as powerful and only
 * legitimate when the direction was chosen before the data were seen; choosing
 * it afterwards is how a p-value of 0.08 becomes one of 0.04.
 */
enum Alternative
{
    case TwoSided;
    case Less;
    case Greater;
}
