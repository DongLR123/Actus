"""Tests for MemoryConfig flush fields (Task 4)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.models.app_config import MemoryConfig


class TestMemoryConfigFlushDefaults:
    def test_flush_enabled_default_false(self) -> None:
        """flush_enabled 默认值为 False。"""
        config = MemoryConfig()
        assert config.flush_enabled is False

    def test_flush_min_steps_default(self) -> None:
        """flush_min_steps 默认值为 2。"""
        config = MemoryConfig()
        assert config.flush_min_steps == 2

    def test_flush_min_new_tokens_default(self) -> None:
        """flush_min_new_tokens 默认值为 3000。"""
        config = MemoryConfig()
        assert config.flush_min_new_tokens == 3000

    def test_flush_max_retries_default(self) -> None:
        """flush_max_retries 默认值为 3。"""
        config = MemoryConfig()
        assert config.flush_max_retries == 3

    def test_flush_circuit_breaker_threshold_default(self) -> None:
        """flush_circuit_breaker_threshold 默认值为 3。"""
        config = MemoryConfig()
        assert config.flush_circuit_breaker_threshold == 3


class TestMemoryConfigFlushCustomValues:
    def test_flush_enabled_can_be_set_true(self) -> None:
        """flush_enabled 可以设置为 True。"""
        config = MemoryConfig(flush_enabled=True)
        assert config.flush_enabled is True

    def test_flush_min_steps_custom(self) -> None:
        """flush_min_steps 可以设置自定义值。"""
        config = MemoryConfig(flush_min_steps=5)
        assert config.flush_min_steps == 5

    def test_flush_min_new_tokens_custom(self) -> None:
        """flush_min_new_tokens 可以设置自定义值。"""
        config = MemoryConfig(flush_min_new_tokens=5000)
        assert config.flush_min_new_tokens == 5000

    def test_flush_max_retries_custom(self) -> None:
        """flush_max_retries 可以设置自定义值。"""
        config = MemoryConfig(flush_max_retries=5)
        assert config.flush_max_retries == 5

    def test_flush_circuit_breaker_threshold_custom(self) -> None:
        """flush_circuit_breaker_threshold 可以设置自定义值。"""
        config = MemoryConfig(flush_circuit_breaker_threshold=7)
        assert config.flush_circuit_breaker_threshold == 7


class TestMemoryConfigFlushValidationRanges:
    def test_flush_min_steps_minimum_boundary(self) -> None:
        """flush_min_steps 最小值为 1。"""
        config = MemoryConfig(flush_min_steps=1)
        assert config.flush_min_steps == 1

    def test_flush_min_steps_maximum_boundary(self) -> None:
        """flush_min_steps 最大值为 10。"""
        config = MemoryConfig(flush_min_steps=10)
        assert config.flush_min_steps == 10

    def test_flush_min_steps_below_minimum_raises(self) -> None:
        """flush_min_steps 低于最小值 1 时抛出 ValidationError。"""
        with pytest.raises(ValidationError):
            MemoryConfig(flush_min_steps=0)

    def test_flush_min_steps_above_maximum_raises(self) -> None:
        """flush_min_steps 高于最大值 10 时抛出 ValidationError。"""
        with pytest.raises(ValidationError):
            MemoryConfig(flush_min_steps=11)

    def test_flush_min_new_tokens_minimum_boundary(self) -> None:
        """flush_min_new_tokens 最小值为 500。"""
        config = MemoryConfig(flush_min_new_tokens=500)
        assert config.flush_min_new_tokens == 500

    def test_flush_min_new_tokens_maximum_boundary(self) -> None:
        """flush_min_new_tokens 最大值为 20000。"""
        config = MemoryConfig(flush_min_new_tokens=20000)
        assert config.flush_min_new_tokens == 20000

    def test_flush_min_new_tokens_below_minimum_raises(self) -> None:
        """flush_min_new_tokens 低于最小值 500 时抛出 ValidationError。"""
        with pytest.raises(ValidationError):
            MemoryConfig(flush_min_new_tokens=499)

    def test_flush_min_new_tokens_above_maximum_raises(self) -> None:
        """flush_min_new_tokens 高于最大值 20000 时抛出 ValidationError。"""
        with pytest.raises(ValidationError):
            MemoryConfig(flush_min_new_tokens=20001)

    def test_flush_max_retries_minimum_boundary(self) -> None:
        """flush_max_retries 最小值为 0。"""
        config = MemoryConfig(flush_max_retries=0)
        assert config.flush_max_retries == 0

    def test_flush_max_retries_maximum_boundary(self) -> None:
        """flush_max_retries 最大值为 10。"""
        config = MemoryConfig(flush_max_retries=10)
        assert config.flush_max_retries == 10

    def test_flush_max_retries_below_minimum_raises(self) -> None:
        """flush_max_retries 低于最小值 0 时抛出 ValidationError。"""
        with pytest.raises(ValidationError):
            MemoryConfig(flush_max_retries=-1)

    def test_flush_max_retries_above_maximum_raises(self) -> None:
        """flush_max_retries 高于最大值 10 时抛出 ValidationError。"""
        with pytest.raises(ValidationError):
            MemoryConfig(flush_max_retries=11)

    def test_flush_circuit_breaker_threshold_minimum_boundary(self) -> None:
        """flush_circuit_breaker_threshold 最小值为 1。"""
        config = MemoryConfig(flush_circuit_breaker_threshold=1)
        assert config.flush_circuit_breaker_threshold == 1

    def test_flush_circuit_breaker_threshold_maximum_boundary(self) -> None:
        """flush_circuit_breaker_threshold 最大值为 10。"""
        config = MemoryConfig(flush_circuit_breaker_threshold=10)
        assert config.flush_circuit_breaker_threshold == 10

    def test_flush_circuit_breaker_threshold_below_minimum_raises(self) -> None:
        """flush_circuit_breaker_threshold 低于最小值 1 时抛出 ValidationError。"""
        with pytest.raises(ValidationError):
            MemoryConfig(flush_circuit_breaker_threshold=0)

    def test_flush_circuit_breaker_threshold_above_maximum_raises(self) -> None:
        """flush_circuit_breaker_threshold 高于最大值 10 时抛出 ValidationError。"""
        with pytest.raises(ValidationError):
            MemoryConfig(flush_circuit_breaker_threshold=11)
