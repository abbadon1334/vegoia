<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit\Graph\Community;

use function file_get_contents;
use function json_decode;

use const JSON_THROW_ON_ERROR;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Graph\Community\Agreement;
use Vegoia\Graph\Partition;
use Vegoia\Tests\Support\Paths;

/**
 * Agreement between two partitions, against scikit-learn.
 *
 * These measures matter for two separate reasons. They are how community
 * detection is evaluated against a ground truth, and they are how you compare
 * one run to the next -- community labels are arbitrary and renumbered on
 * every run, so "did the partition change" cannot be answered by comparing
 * labels and needs a measure that ignores them.
 *
 * The conventions are not universal. NMI can normalise by the arithmetic
 * mean, geometric mean, min or max of the two entropies, and the four
 * disagree; sklearn's default is arithmetic, which is what is implemented and
 * pinned here.
 */
#[CoversClass(Agreement::class)]
final class AgreementTest extends TestCase
{
    /** @return iterable<string, array{string}> */
    public static function cases(): iterable
    {
        /** @var array{cases: array<string, array{a: list<int>, b: list<int>, nmi: float, ari: float}>} $data */
        $data = json_decode(
            (string) file_get_contents(Paths::fixture('partition_agreement.json')),
            associative: true,
            flags: JSON_THROW_ON_ERROR,
        );

        foreach ($data['cases'] as $name => $_) {
            yield $name => [$name];
        }
    }

    #[DataProvider('cases')]
    public function test_normalised_mutual_information_matches_scikit_learn(string $case): void
    {
        [$a, $b, $expected] = self::load($case, 'nmi');

        self::assertEqualsWithDelta(
            $expected,
            Agreement::normalisedMutualInformation($a, $b),
            1.0e-12,
            "{$case}: NMI",
        );
    }

    #[DataProvider('cases')]
    public function test_adjusted_rand_index_matches_scikit_learn(string $case): void
    {
        [$a, $b, $expected] = self::load($case, 'ari');

        self::assertEqualsWithDelta(
            $expected,
            Agreement::adjustedRandIndex($a, $b),
            1.0e-12,
            "{$case}: ARI",
        );
    }

    /**
     * Community labels are arbitrary, so a measure that changed when they were
     * renumbered would be useless for the job it exists to do.
     */
    public function test_both_measures_ignore_how_communities_are_numbered(): void
    {
        $a = Partition::fromMembership([0, 0, 1, 1, 2]);
        $b = Partition::fromMembership([7, 7, 3, 3, 9]);

        self::assertSame(1.0, Agreement::normalisedMutualInformation($a, $b));
        self::assertSame(1.0, Agreement::adjustedRandIndex($a, $b));
    }

    public function test_the_measures_are_symmetric(): void
    {
        $a = Partition::fromMembership([0, 0, 0, 1, 1, 2]);
        $b = Partition::fromMembership([0, 0, 1, 1, 2, 2]);

        self::assertEqualsWithDelta(
            Agreement::normalisedMutualInformation($a, $b),
            Agreement::normalisedMutualInformation($b, $a),
            1.0e-15,
        );
        self::assertEqualsWithDelta(
            Agreement::adjustedRandIndex($a, $b),
            Agreement::adjustedRandIndex($b, $a),
            1.0e-15,
        );
    }

    /**
     * ARI is corrected for chance and so can go negative -- two partitions can
     * agree less than random labelling would. NMI cannot.
     */
    public function test_adjusted_rand_index_can_be_negative_where_nmi_cannot(): void
    {
        $a = Partition::fromMembership([0, 0, 1, 1]);
        $b = Partition::fromMembership([0, 1, 0, 1]);

        // -0.5 exactly in real arithmetic; the last bit is not reachable
        // through the chance-correction division.
        self::assertEqualsWithDelta(-0.5, Agreement::adjustedRandIndex($a, $b), 1.0e-15);
        self::assertSame(0.0, Agreement::normalisedMutualInformation($a, $b));
    }

    /**
     * Variation of information is a true metric: non-negative, zero only for
     * identical partitions, and it satisfies the triangle inequality -- which
     * NMI and ARI do not, so distances between partitions cannot be built from
     * them.
     */
    public function test_variation_of_information_is_zero_only_for_identical_partitions(): void
    {
        $a = Partition::fromMembership([0, 0, 1, 1]);

        self::assertSame(0.0, Agreement::variationOfInformation($a, $a));
        self::assertSame(0.0, Agreement::variationOfInformation($a, Partition::fromMembership([4, 4, 8, 8])));
        self::assertGreaterThan(0.0, Agreement::variationOfInformation($a, Partition::fromMembership([0, 1, 0, 1])));
    }

    public function test_variation_of_information_obeys_the_triangle_inequality(): void
    {
        $a = Partition::fromMembership([0, 0, 0, 1, 1, 1]);
        $b = Partition::fromMembership([0, 0, 1, 1, 2, 2]);
        $c = Partition::fromMembership([0, 1, 2, 3, 4, 5]);

        self::assertLessThanOrEqual(
            Agreement::variationOfInformation($a, $b) + Agreement::variationOfInformation($b, $c) + 1.0e-12,
            Agreement::variationOfInformation($a, $c),
        );
    }

    public function test_it_refuses_partitions_of_different_sizes(): void
    {
        $this->expectException(InvalidArgument::class);

        Agreement::normalisedMutualInformation(
            Partition::fromMembership([0, 0, 1]),
            Partition::fromMembership([0, 1]),
        );
    }

    public function test_empty_partitions_agree_completely(): void
    {
        $empty = Partition::fromMembership([]);

        self::assertSame(1.0, Agreement::normalisedMutualInformation($empty, $empty));
        self::assertSame(1.0, Agreement::adjustedRandIndex($empty, $empty));
        self::assertSame(0.0, Agreement::variationOfInformation($empty, $empty));
    }

    /** @return array{Partition, Partition, float} */
    private static function load(string $case, string $measure): array
    {
        /** @var array{cases: array<string, array{a: list<int>, b: list<int>, nmi: float, ari: float}>} $data */
        $data = json_decode(
            (string) file_get_contents(Paths::fixture('partition_agreement.json')),
            associative: true,
            flags: JSON_THROW_ON_ERROR,
        );

        $entry = $data['cases'][$case];

        /** @var float $expected */
        $expected = $entry[$measure];

        return [
            Partition::fromMembership($entry['a']),
            Partition::fromMembership($entry['b']),
            $expected,
        ];
    }
}
