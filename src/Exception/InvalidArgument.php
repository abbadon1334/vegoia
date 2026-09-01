<?php

declare(strict_types=1);

namespace Vegoia\Exception;

use InvalidArgumentException;

use function is_nan;
use function sprintf;

final class InvalidArgument extends InvalidArgumentException implements VegoiaException
{
    public static function emptyDataset(string $operation): self
    {
        return new self("Cannot compute {$operation} of an empty dataset.");
    }

    public static function tooFewValues(string $operation, int $given, int $required): self
    {
        return new self("{$operation} needs at least {$required} values, {$given} given.");
    }

    public static function nodeOutOfRange(int $node, int $order): self
    {
        return new self(
            $order === 0
                ? "Node {$node} does not exist: the graph has no nodes."
                : "Node {$node} does not exist: the graph has nodes 0..".($order - 1).'.'
        );
    }

    public static function negativeOrder(int $order): self
    {
        return new self("A graph cannot have {$order} nodes.");
    }

    public static function malformedEdge(string $detail): self
    {
        return new self("Malformed edge: {$detail}");
    }

    /**
     * @param string $because what the directed case would need instead, so the
     *                        message says why rather than only that
     */
    public static function directedNotSupported(string $operation, string $because): self
    {
        return new self("{$operation} is defined for undirected graphs only. {$because}");
    }

    public static function outOfRange(string $what, float $given, float $low, float $high): self
    {
        // PHP 8.5 warns when NAN is coerced to a string, and a message that
        // emits a warning while reporting an error is not much of a message.
        // A NAN is better described by notANumber() anyway; this only keeps
        // the general constructor safe for callers that have not separated
        // the two cases.
        return new self(sprintf(
            '%s must lie in [%s, %s], %s given.',
            $what,
            self::describe($low),
            self::describe($high),
            self::describe($given),
        ));
    }

    /**
     * Not out of range: not a number at all.
     *
     * Worth its own constructor because the two are different failures and
     * only one of them is about bounds. NAN reaches this library from its own
     * output -- Fit::pValue() returns it for a zero coefficient with a zero
     * standard error -- so it arrives through ordinary use rather than through
     * abuse.
     */
    public static function notANumber(string $what): self
    {
        return new self("{$what} is not a number.");
    }

    private static function describe(float $value): string
    {
        return match (true) {
            is_nan($value) => 'NAN',
            $value === INF => 'INF',
            $value === -INF => '-INF',
            default => (string) $value,
        };
    }

    /** A special function asked for a value outside the region where it is defined. */
    public static function outOfDomain(string $what, float $given, string $requirement): self
    {
        return new self("{$what} is undefined for {$given}: {$requirement}.");
    }
}
