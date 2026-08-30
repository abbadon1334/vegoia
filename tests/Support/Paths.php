<?php

declare(strict_types=1);

namespace Vegoia\Tests\Support;

use function dirname;

final class Paths
{
    public static function root(): string
    {
        return dirname(__DIR__, 2);
    }

    public static function fixture(string $relative): string
    {
        return self::root() . '/resources/fixtures/' . $relative;
    }
}
