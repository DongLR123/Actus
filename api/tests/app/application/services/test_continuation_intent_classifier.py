from __future__ import annotations

import asyncio

import pytest
from app.application.services.continuation_intent_classifier import (
    ContinuationIntentClassifier,
)


pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    def __init__(self, content: str, delay_seconds: float = 0.0) -> None:
        self._content = content
        self._delay_seconds = delay_seconds

    async def ainvoke(self, messages, **kwargs):  # noqa: ANN001, ANN003
        if self._delay_seconds > 0:
            await asyncio.sleep(self._delay_seconds)
        return _FakeMessage(self._content)


async def test_classifier_parses_true_false_payload() -> None:
    classifier_true = ContinuationIntentClassifier(
        llm=_FakeLLM('{"is_continuation": true}'),
        timeout_seconds=1.0,
    )
    classifier_false = ContinuationIntentClassifier(
        llm=_FakeLLM('{"is_continuation": false}'),
        timeout_seconds=1.0,
    )

    assert (
        await classifier_true.classify(
            current_message="好的，继续",
            previous_substantive_message="请帮我优化 SQL 查询",
        )
        is True
    )
    assert (
        await classifier_false.classify(
            current_message="sql优化",
            previous_substantive_message="请帮我优化 SQL 查询",
        )
        is False
    )


async def test_classifier_returns_false_when_json_invalid() -> None:
    classifier = ContinuationIntentClassifier(
        llm=_FakeLLM("not-json"),
        timeout_seconds=1.0,
    )

    assert (
        await classifier.classify(
            current_message="好的，继续",
            previous_substantive_message="请帮我优化 SQL 查询",
        )
        is False
    )


async def test_classifier_returns_false_when_timeout() -> None:
    classifier = ContinuationIntentClassifier(
        llm=_FakeLLM('{"is_continuation": true}', delay_seconds=0.2),
        timeout_seconds=0.05,
    )

    assert (
        await classifier.classify(
            current_message="好的，继续",
            previous_substantive_message="请帮我优化 SQL 查询",
        )
        is False
    )
