<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit\Stats;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;
use Vegoia\Stats\Correlation;
use Vegoia\Stats\Ranks;

/**
 * What counts as a tie, and the defect that made two answers to that question
 * live in one file.
 */
#[CoversClass(Ranks::class)]
final class RanksTest extends TestCase
{
    /**
     * Two doubles that PHP's string cast conflates at its default precision.
     *
     * This is the pair that broke Kendall's tau-b. Nothing about it is
     * contrived -- it is 0.1 and a few units in the last place above it, the
     * kind of thing any arithmetic produces.
     */
    private const float NEARLY_A_TENTH = 0.10000000000000012;

    public function test_the_two_values_this_class_exists_for_really_do_collide(): void
    {
        // If PHP ever changes its default precision this stops being the right
        // example, and the test says so rather than quietly passing for a new
        // reason.
        self::assertSame(
            (string) 0.1,
            (string) self::NEARLY_A_TENTH,
            'the two values no longer collide as strings; find another pair or drop the guard',
        );

        self::assertNotSame(0.1, self::NEARLY_A_TENTH, 'they are distinct doubles');
    }

    public function test_values_that_only_collide_as_strings_are_not_tied(): void
    {
        self::assertSame([1, 1], Ranks::tieSizes([0.1, self::NEARLY_A_TENTH]));
        self::assertSame(0.0, Ranks::tiedPairs([0.1, self::NEARLY_A_TENTH]));
        self::assertSame([1.0, 2.0], Ranks::midranks([0.1, self::NEARLY_A_TENTH]));
    }

    public function test_values_that_are_the_same_double_are_tied(): void
    {
        self::assertSame([2], Ranks::tieSizes([0.1, 0.1]));
        self::assertSame(1.0, Ranks::tiedPairs([0.1, 0.1]));
        self::assertSame([1.5, 1.5], Ranks::midranks([0.1, 0.1]));
    }

    /**
     * The property every rank statistic downstream relies on: averaging ties
     * leaves the total unchanged, so the null mean of a rank sum stays in
     * closed form.
     *
     * @param list<float> $values
     */
    #[DataProvider('samples')]
    public function test_the_ranks_always_sum_to_n_times_n_plus_one_over_two(array $values): void
    {
        $n = count($values);
        $total = array_sum(Ranks::midranks($values));

        self::assertEqualsWithDelta($n * ($n + 1) / 2.0, $total, 1.0e-12);
    }

    /**
     * The group sizes always account for every observation.
     *
     * @param list<float> $values
     */
    #[DataProvider('samples')]
    public function test_the_tie_sizes_account_for_every_value(array $values): void
    {
        self::assertSame(count($values), array_sum(Ranks::tieSizes($values)));
    }

    /** @return iterable<string, array{list<float>}> */
    public static function samples(): iterable
    {
        yield 'distinct' => [[3.0, 1.0, 2.0]];
        yield 'all tied' => [[7.0, 7.0, 7.0, 7.0]];
        yield 'one tied group' => [[1.0, 2.0, 2.0, 2.0, 5.0]];
        yield 'several groups' => [[1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0]];
        yield 'colliding as strings' => [[0.1, self::NEARLY_A_TENTH, 0.2]];
        yield 'negative and zero' => [[-1.0, 0.0, -1.0, 0.0]];
        yield 'one value' => [[42.0]];
        yield 'empty' => [[]];
    }

    /** Ranks come back parallel to the input, not to the sorted order. */
    public function test_the_ranks_line_up_with_the_input(): void
    {
        self::assertSame([3.0, 1.0, 2.0], Ranks::midranks([30.0, 10.0, 20.0]));
    }

    /** Group sizes come back in ascending order of the value, not of size. */
    public function test_the_tie_sizes_are_in_value_order(): void
    {
        // Sorted, that is [1, 2, 2, 3, 3, 3]: one, then two, then three.
        self::assertSame([1, 2, 3], Ranks::tieSizes([2.0, 2.0, 3.0, 3.0, 3.0, 1.0]));
    }

    /**
     * The end-to-end consequence, on the case that exposed the defect.
     *
     * Kendall's tau-b needs its numerator and denominator to agree about the
     * pair. Before the extraction they did not, and the answer was neither the
     * tied one nor the untied one.
     */
    public function test_kendall_agrees_with_itself_about_what_a_tie_is(): void
    {
        $x = [0.1, self::NEARLY_A_TENTH, 0.2, 0.3, 0.4, 0.5];
        $y = [1.0, 2.0, 3.0, 6.0, 4.0, 5.0];

        // scipy, on the same input: the pair is not tied, so tau-b is 11/15.
        self::assertEqualsWithDelta(0.7333333333333333, Correlation::kendall($x, $y), 1.0e-12);

        // And had it genuinely been tied, the answer would be this instead --
        // pinned so the number above cannot be reached by accident from the
        // wrong branch.
        $tied = [0.1, 0.1, 0.2, 0.3, 0.4, 0.5];
        self::assertEqualsWithDelta(0.6900655593423541, Correlation::kendall($tied, $y), 1.0e-12);
    }
}
