"""Tests for retry utilities with exponential backoff."""

from __future__ import annotations

import time

import pytest

from ragflow_orchestrator.retry import retry, retry_or_log


class TestRetry:
    """Test retry decorator with exponential backoff."""

    def test_retry_succeeds_on_first_attempt(self) -> None:
        @retry(max_retries=2, initial_delay=0.01)
        def successful_func() -> str:
            return "success"

        result = successful_func()
        assert result == "success"

    def test_retry_succeeds_after_retries(self) -> None:
        call_count = 0

        @retry(max_retries=2, initial_delay=0.01)
        def eventually_succeeds() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("temporary failure")
            return "success"

        result = eventually_succeeds()
        assert result == "success"
        assert call_count == 2

    def test_retry_exhausts_attempts(self) -> None:
        call_count = 0

        @retry(max_retries=2, initial_delay=0.01)
        def always_fails() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("always fails")

        with pytest.raises(ValueError, match="always fails"):
            always_fails()
        assert call_count == 3  # initial + 2 retries

    def test_retry_respects_max_retries(self) -> None:
        call_count = 0

        @retry(max_retries=1, initial_delay=0.01)
        def fails_twice() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("fails")

        with pytest.raises(ValueError):
            fails_twice()
        assert call_count == 2  # initial + 1 retry

    def test_retry_with_jitter(self) -> None:
        @retry(max_retries=1, initial_delay=0.01, jitter=True)
        def successful_func() -> str:
            return "success"

        result = successful_func()
        assert result == "success"

    def test_retry_with_max_delay(self) -> None:
        """Test that delay is capped at max_delay."""
        call_count = 0

        @retry(max_retries=3, initial_delay=10.0, max_delay=0.05, backoff_factor=2.0, jitter=False)
        def fails_multiple_times() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise ValueError("fails")
            return "success"

        start = time.time()
        fails_multiple_times()
        elapsed = time.time() - start

        # Should have 4 calls (1 initial + 3 retries)
        # With max_delay=0.05, total sleep should be <= 0.05 * 3
        assert call_count == 4
        assert elapsed < 0.2  # generous upper bound due to system variability

    def test_retry_with_backoff_factor(self) -> None:
        """Test that backoff increases delay exponentially."""
        call_count = 0

        @retry(max_retries=2, initial_delay=0.01, max_delay=1.0, backoff_factor=2.0, jitter=False)
        def fails_twice() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("fails")
            return "success"

        start = time.time()
        result = fails_twice()
        elapsed = time.time() - start

        # Delays: 0.01, 0.02 = 0.03 total
        assert result == "success"
        assert elapsed >= 0.03


class TestRetryOrLog:
    """Test retry_or_log decorator with graceful fallback."""

    def test_retry_or_log_succeeds_on_first_attempt(self) -> None:
        @retry_or_log(max_retries=2, initial_delay=0.01)
        def successful_func() -> str:
            return "success"

        result = successful_func()
        assert result == "success"

    def test_retry_or_log_succeeds_after_retries(self) -> None:
        call_count = 0

        @retry_or_log(max_retries=2, initial_delay=0.01)
        def eventually_succeeds() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("temporary failure")
            return "success"

        result = eventually_succeeds()
        assert result == "success"
        assert call_count == 2

    def test_retry_or_log_returns_none_on_exhaustion(self) -> None:
        call_count = 0

        @retry_or_log(max_retries=2, initial_delay=0.01)
        def always_fails() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("always fails")

        result = always_fails()
        assert result is None
        assert call_count == 3  # initial + 2 retries

    def test_retry_or_log_with_on_failure_callback(self) -> None:
        callback_called = False
        callback_exc: Exception | None = None
        callback_msg: str | None = None

        def on_failure(exc: Exception, msg: str) -> None:
            nonlocal callback_called, callback_exc, callback_msg
            callback_called = True
            callback_exc = exc
            callback_msg = msg

        @retry_or_log(max_retries=1, initial_delay=0.01, on_failure=on_failure)
        def always_fails() -> str:
            raise ValueError("test failure")

        result = always_fails()
        assert result is None
        assert callback_called
        assert isinstance(callback_exc, ValueError)
        assert callback_msg is not None
        assert "test failure" in callback_msg
