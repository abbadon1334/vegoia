<?php

declare(strict_types=1);

namespace Vegoia\Stats;

/**
 * How much arithmetic care a statistic is worth.
 *
 * The default is Extended, and on most data it costs speed for nothing: a
 * plain summation already reaches thirteen correct digits on NIST's PiDigits.
 * What it buys is the tail. On the datasets NIST built to break statistical
 * software -- large values differing in their last places -- plain summation
 * falls to nine digits on the lag-1 autocorrelation of NumAcc4, while extended
 * reaches 15.65. You cannot tell which kind of data you have by looking at the
 * answer, which is why accuracy is the default and speed is the choice.
 *
 * Measured on 5000 values:
 *
 *                        mean      autocorrelation
 *     Fast             0.029 ms         0.18 ms
 *     Extended         0.310 ms         3.00 ms
 *
 * So Extended costs roughly 10x on a mean and 16x on an autocorrelation. That
 * is worth paying once per series and not worth paying inside a loop over
 * thousands of them -- which is exactly when Fast is the right answer.
 *
 * Fast is not sloppy. It is compensated summation's absence, not carelessness:
 * ordinary floating point, the same as numpy's, and on well-conditioned data
 * it lands within a digit of Extended. Two of the NIST datasets it even beats
 * numpy on.
 */
enum Precision
{
    /**
     * Compensated summation, extended-precision division, exact products.
     * Accurate to the limit of what binary64 inputs allow.
     */
    case Extended;

    /**
     * Ordinary floating point. Around ten times faster, and around two digits
     * less accurate on well-conditioned data -- far more on data built to
     * expose the difference.
     */
    case Fast;
}
