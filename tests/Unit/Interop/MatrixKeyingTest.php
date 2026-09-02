<?php

declare(strict_types=1);

namespace Vegoia\Tests\Unit\Interop;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;
use Vegoia\Exception\InvalidArgument;
use Vegoia\Interop\LabelledGraph;

/**
 * An adjacency matrix is its rows and columns, not their keys.
 *
 * fromMatrix used a row's key as a node index and a label's key as a node
 * number, which is right for a list and wrong for anything else -- and wrong
 * quietly. A matrix keyed 2, 5, 9 came back as a graph with no edges at all,
 * because every column index fell below its row index and was skipped as the
 * lower triangle. No warning, no exception: a populated matrix in, an empty
 * graph out.
 *
 * Labels failed the same way in the other direction. Given ['a', 'b', 'c']
 * under keys 3, 7, 11 the count check passed, every lookup missed, and the
 * nodes came back named 0, 1, 2 -- labels supplied and silently discarded.
 *
 * The same defect as in Similarity and in Correlation before it, and fixed
 * the same way: position is the meaning, so both are read by position.
 */
#[CoversClass(LabelledGraph::class)]
final class MatrixKeyingTest extends TestCase
{
    /** @return list<list<float>> a path a-b-c, the b-c edge weighted 2 */
    private static function matrix(): array
    {
        return [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 2.0],
            [0.0, 2.0, 0.0],
        ];
    }

    /** @return list<string> */
    private static function edges(LabelledGraph $graph): array
    {
        $out = [];

        foreach ($graph->graph->edges() as [$from, $to, $weight]) {
            $out[] = sprintf('%s-%s:%s', $graph->identifier($from), $graph->identifier($to), $weight);
        }

        return $out;
    }

    /** @return iterable<string, array{array<array-key, array<array-key, float>>}> */
    public static function keyings(): iterable
    {
        yield 'a list of lists, as documented' => [self::matrix()];

        // Rows keyed by anything other than 0..n-1 used to yield no edges.
        yield 'rows with gaps' => [[2 => self::matrix()[0], 5 => self::matrix()[1], 9 => self::matrix()[2]]];

        yield 'rows in descending keys' => [
            [2 => self::matrix()[0], 1 => self::matrix()[1], 0 => self::matrix()[2]],
        ];

        yield 'columns with gaps' => [
            array_map(
                static fn (array $row): array => [4 => $row[0], 8 => $row[1], 12 => $row[2]],
                self::matrix(),
            ),
        ];

        yield 'both with gaps' => [
            [
                2 => [4 => 0.0, 8 => 1.0, 12 => 0.0],
                5 => [4 => 1.0, 8 => 0.0, 12 => 2.0],
                9 => [4 => 0.0, 8 => 2.0, 12 => 0.0],
            ],
        ];
    }

    /** @param array<array-key, array<array-key, float>> $matrix */
    #[DataProvider('keyings')]
    public function test_the_same_matrix_builds_the_same_graph_however_it_is_keyed(array $matrix): void
    {
        $graph = LabelledGraph::fromMatrix($matrix, ['a', 'b', 'c']);

        self::assertSame(3, $graph->graph->order());
        self::assertSame(2, $graph->graph->size(), 'a matrix with two edges must produce two edges');
        self::assertSame(['a-b:1', 'b-c:2'], self::edges($graph));
    }

    /** @return iterable<string, array{array<array-key, string>}> */
    public static function labelKeyings(): iterable
    {
        yield 'a list' => [['a', 'b', 'c']];
        yield 'gaps' => [[3 => 'a', 7 => 'b', 11 => 'c']];
        yield 'descending' => [[2 => 'a', 1 => 'b', 0 => 'c']];
        yield 'string keys' => [['first' => 'a', 'second' => 'b', 'third' => 'c']];
    }

    /** @param array<array-key, string> $labels */
    #[DataProvider('labelKeyings')]
    public function test_labels_are_read_in_order_however_they_are_keyed(array $labels): void
    {
        self::assertSame(
            ['a-b:1', 'b-c:2'],
            self::edges(LabelledGraph::fromMatrix(self::matrix(), $labels)),
        );
    }

    /**
     * A matrix that is not square is still refused, and the message still
     * names a row a caller can find.
     */
    public function test_a_ragged_matrix_is_refused(): void
    {
        $this->expectException(InvalidArgument::class);
        $this->expectExceptionMessageMatches('/Row 1 has 2 entries, not 3/');

        LabelledGraph::fromMatrix([[0.0, 1.0, 0.0], [1.0, 0.0], [0.0, 0.0, 0.0]]);
    }

    /** The wrong number of labels is refused whatever the keys are. */
    public function test_the_wrong_number_of_labels_is_refused(): void
    {
        $this->expectException(InvalidArgument::class);
        $this->expectExceptionMessageMatches('/3 rows and 2 labels/');

        LabelledGraph::fromMatrix(self::matrix(), [5 => 'a', 9 => 'b']);
    }

    /**
     * Directed matrices read both triangles, so the keying has to survive
     * there too -- and a directed matrix is where an asymmetry would show a
     * transposition rather than hide it.
     */
    public function test_a_directed_matrix_keeps_its_direction(): void
    {
        $matrix = [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [3.0, 0.0, 0.0],
        ];

        $expected = ['a-b:1', 'b-c:2', 'c-a:3'];

        self::assertSame(
            $expected,
            self::edges(LabelledGraph::fromMatrix($matrix, ['a', 'b', 'c'], directed: true)),
        );

        self::assertSame(
            $expected,
            self::edges(LabelledGraph::fromMatrix(
                [4 => $matrix[0], 8 => $matrix[1], 12 => $matrix[2]],
                [3 => 'a', 7 => 'b', 11 => 'c'],
                directed: true,
            )),
        );
    }
}
