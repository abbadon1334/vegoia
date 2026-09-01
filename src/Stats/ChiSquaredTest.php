<?php

declare(strict_types=1);

namespace Vegoia\Stats;

use function abs;
use function count;
use function min;

use Vegoia\Exception\InvalidArgument;
use Vegoia\Stats\Distribution\ChiSquared;
use Vegoia\Support\CompensatedSum;

/**
 * Are the rows and the columns of a contingency table independent?
 *
 * The test for "does this categorical thing go with that one" -- whether the
 * documents a community detection put together share a source, whether an
 * error rate depends on which parser produced the input. It compares what was
 * counted against what independence would have predicted from the margins
 * alone.
 */
final readonly class ChiSquaredTest
{
    /** @param list<list<float>> $expected */
    private function __construct(
        public float $statistic,
        public int $degreesOfFreedom,
        public array $expected,
        public Continuity $continuity,
        public int $rows,
        public int $columns,
        public float $total,
    ) {
    }

    /**
     * @param list<list<int|float>> $table observed counts, one list per row
     */
    public static function independence(
        array $table,
        Continuity $continuity = Continuity::Corrected,
    ): self {
        $rows = count($table);

        if ($rows < 2) {
            throw InvalidArgument::tooFewValues('A contingency table needs rows; it', $rows, 2);
        }

        $columns = count($table[0]);

        if ($columns < 2) {
            throw InvalidArgument::tooFewValues('A contingency table needs columns; it', $columns, 2);
        }

        $rowTotals = [];
        $columnTotals = array_fill(0, $columns, 0.0);
        $total = 0.0;

        foreach ($table as $row => $counts) {
            if (count($counts) !== $columns) {
                throw InvalidArgument::malformedEdge(
                    "Row {$row} has " . count($counts) . " entries, not {$columns}"
                );
            }

            $rowTotal = 0.0;

            foreach ($counts as $column => $count) {
                if ($count < 0) {
                    throw InvalidArgument::outOfRange(
                        "The count at ({$row}, {$column})",
                        (float) $count,
                        0.0,
                        INF,
                    );
                }

                $rowTotal += $count;
                $columnTotals[$column] += $count;
            }

            $rowTotals[] = $rowTotal;
            $total += $rowTotal;
        }

        if ($total <= 0.0) {
            throw InvalidArgument::emptyDataset('a contingency table with no observations');
        }

        $degreesOfFreedom = ($rows - 1) * ($columns - 1);

        // Yates only touches a table with one degree of freedom. On anything
        // larger the half-step has no justification and applying it anyway is
        // a well-known way to report a statistic nobody else gets.
        $correcting = $continuity === Continuity::Corrected && $degreesOfFreedom === 1;

        $expected = [];
        $statistic = new CompensatedSum();

        foreach ($table as $row => $counts) {
            $expectedRow = [];

            foreach ($counts as $column => $count) {
                $predicted = $rowTotals[$row] * $columnTotals[$column] / $total;

                if ($predicted <= 0.0) {
                    throw InvalidArgument::malformedEdge(
                        "Row {$row} or column {$column} is entirely zero, which leaves a cell with "
                        . 'zero expected count and the statistic divides by it'
                    );
                }

                $expectedRow[] = $predicted;

                $difference = (float) $count - $predicted;

                if ($correcting) {
                    // Shift the observation towards the expectation by half a
                    // step -- but never past it. The textbook
                    // (|o - e| - 0.5)^2 / e overshoots when the difference is
                    // already below half a step and reports a positive
                    // statistic where the answer is zero: measured on a table
                    // whose every difference is 0.244, it gives 0.0232 against
                    // the correct 0. SciPy carries this as a fixed bug.
                    // Signum times the clamped shift. When the difference is
                    // already zero the signum is zero and nothing moves, so
                    // there is no special case to write.
                    $difference -= ($difference <=> 0.0) * min(0.5, abs($difference));
                }

                // Cell by cell, never the algebraically equal sum(o^2/e) - N.
                // That shortcut is the one-pass variance in a different
                // costume: it differences two large near-equal quantities, and
                // on a large table with a good fit the answer cancels away
                // entirely. OneWayAnova's docblock makes the same argument.
                $statistic->add($difference * $difference / $predicted);
            }

            /** @var list<float> $expectedRow */
            $expected[] = $expectedRow;
        }

        /** @var list<list<float>> $expected */
        return new self(
            $statistic->value(),
            $degreesOfFreedom,
            $expected,
            $continuity,
            $rows,
            $columns,
            $total,
        );
    }

    /**
     * How often a table this far from independence arises when the two
     * classifications are unrelated.
     *
     * The upper tail, computed directly. Only large values of the statistic
     * are evidence against independence, so the test is one-sided by
     * construction.
     */
    public function pValue(): float
    {
        return new ChiSquared((float) $this->degreesOfFreedom)->survival($this->statistic);
    }
}
