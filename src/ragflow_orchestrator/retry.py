"""Retry utilities with exponential backoff for resilient HTTP and operation handling."""

from __future__ import annotations

import logging
import random
import time
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 32.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (total attempts = max_retries + 1).
        initial_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay in seconds between retries.
        backoff_factor: Multiplier for delay after each failure.
        jitter: Whether to add random jitter to delay.

    Example:
        @retry(max_retries=3, initial_delay=1.0)
        def fetch_url(url: str) -> str:
            ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            delay = initial_delay
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exception = exc
                    if attempt < max_retries:
                        if jitter:
                            jittered_delay = delay * (0.5 + random.random())
                        else:
                            jittered_delay = delay
                        jittered_delay = min(jittered_delay, max_delay)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {exc}. "
                            f"Retrying in {jittered_delay:.2f}s..."
                        )
                        time.sleep(jittered_delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}: {exc}")

            raise last_exception

        return wrapper

    return decorator


def retry_or_log(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 32.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    on_failure: Callable[[Exception, str], None] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T | None]]:
    """
    Decorator for retrying functions with exponential backoff, returning None on final failure.

    Instead of raising an exception, logs the error and returns None after all retries are exhausted.
    Useful for gracefully handling non-critical operations.

    Args:
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay in seconds between retries.
        backoff_factor: Multiplier for delay after each failure.
        jitter: Whether to add random jitter to delay.
        on_failure: Optional callback to handle failure (func, error_msg) -> None.

    Example:
        @retry_or_log(max_retries=2)
        def fetch_url(url: str) -> str | None:
            ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T | None]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | None:
            delay = initial_delay
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exception = exc
                    if attempt < max_retries:
                        if jitter:
                            jittered_delay = delay * (0.5 + random.random())
                        else:
                            jittered_delay = delay
                        jittered_delay = min(jittered_delay, max_delay)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {exc}. "
                            f"Retrying in {jittered_delay:.2f}s..."
                        )
                        time.sleep(jittered_delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        error_msg = f"All {max_retries + 1} attempts failed for {func.__name__}: {exc}"
                        logger.error(error_msg)
                        if on_failure:
                            on_failure(exc, error_msg)

            return None

        return wrapper

    return decorator
