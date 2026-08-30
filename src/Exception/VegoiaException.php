<?php

declare(strict_types=1);

namespace Vegoia\Exception;

use Throwable;

/**
 * Every exception this library throws implements this, so callers can catch
 * "something Vegoia raised" without enumerating concrete classes.
 */
interface VegoiaException extends Throwable
{
}
