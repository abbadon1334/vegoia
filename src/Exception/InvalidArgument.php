<?php

declare(strict_types=1);

namespace Vegoia\Exception;

use InvalidArgumentException;

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
        return new self("{$what} must lie in [{$low}, {$high}], {$given} given.");
    }

    /** A special function asked for a value outside the region where it is defined. */
    public static function outOfDomain(string $what, float $given, string $requirement): self
    {
        return new self("{$what} is undefined for {$given}: {$requirement}.");
    }
}
