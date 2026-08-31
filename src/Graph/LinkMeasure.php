<?php

declare(strict_types=1);

namespace Vegoia\Graph;

/**
 * How to score a pair of nodes for a link that is not there.
 *
 * The five disagree on purpose, and choosing between them is choosing what
 * you think an edge means.
 *
 * CommonNeighbours counts and nothing else. It is the baseline the others are
 * measured against, and it is biased towards hubs: two nodes of degree 200
 * will share more neighbours than two of degree 5 whatever the truth is.
 *
 * Jaccard divides that count by the size of the union, so it asks what
 * proportion of the pair's world is shared rather than how much of it. Two
 * obscure nodes sharing both their neighbours beat two hubs sharing three of
 * four hundred.
 *
 * AdamicAdar and ResourceAllocation both weight each shared neighbour by how
 * unusual it is -- by 1/log(degree) and 1/degree. Sharing a neighbour that
 * everybody has says little; sharing one that almost nobody has says a lot.
 * The second punishes hubs harder, and the two rarely disagree on the order,
 * only on the spread.
 *
 * PreferentialAttachment ignores shared neighbours entirely and multiplies the
 * degrees. It is the odd one out and worth keeping: it encodes "busy nodes get
 * busier", which is true of citation and social graphs and false of most
 * others, and it will happily rank a pair with nothing whatever in common.
 */
enum LinkMeasure
{
    case CommonNeighbours;
    case Jaccard;
    case AdamicAdar;
    case ResourceAllocation;
    case PreferentialAttachment;

    /**
     * Whether a pair with no shared neighbour can still score above zero.
     *
     * Only preferential attachment can, since it never looks at the
     * neighbourhood. It is what decides whether ranking may stop at the
     * two-hop candidates or has to consider every node in the graph.
     */
    public function scoresStrangers(): bool
    {
        return $this === self::PreferentialAttachment;
    }
}
