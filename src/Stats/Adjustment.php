<?php

declare(strict_types=1);

namespace Vegoia\Stats;

/**
 * Which error rate a family of p-values is being held to.
 *
 * Not a matter of taste, and the three are not interchangeable. Bonferroni and
 * Holm control the family-wise error rate -- the chance of even one false
 * positive anywhere in the family. Benjamini-Hochberg controls the false
 * discovery rate, the expected share of the rejections that are wrong. On
 * twenty tests at the five per cent level the first two are asking for at most
 * one mistake in the whole family; the third is content with one in every
 * twenty findings. Those are different experiments, and choosing between them
 * is the question rather than a detail of it -- which is why nothing here has
 * a default.
 *
 * Holm dominates Bonferroni: it rejects everything Bonferroni rejects and
 * sometimes more, at the same guarantee, so there is no situation in which
 * Bonferroni is the better answer. It is here because reviewers ask for it by
 * name and because it is the one anybody can check by hand.
 */
enum Adjustment
{
    /** Every p-value multiplied by the size of the family. */
    case Bonferroni;

    /** Bonferroni applied step-down, from the smallest p-value outward. */
    case Holm;

    /** Benjamini-Hochberg step-up, controlling the false discovery rate. */
    case BenjaminiHochberg;
}
