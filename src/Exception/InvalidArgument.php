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

    public static function outOfRange(string $what, float $given, float $low, float $high): self
    {
        return new self("{$what} must lie in [{$low}, {$high}], {$given} given.");
    }
}
