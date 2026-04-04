"""Tests for AudioFileProcessor — dual engine: OpenAI API + sandbox faster-whisper."""
from __future__ import annotations

import asyncio
import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.domain.external.file_processor import FileProcessResult
from app.domain.models.app_config import AudioProcessorConfig
from app.infrastructure.external.file_processors.audio import AudioFileProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kwargs) -> AudioProcessorConfig:
    defaults = {
        "provider": "sandbox_whisper",
        "openai_api_key": "sk-test",
        "openai_base_url": "https://api.openai.com/v1",
        "openai_model": "whisper-1",
    }
    defaults.update(kwargs)
    return AudioProcessorConfig(**defaults)


def _make_exec_result(returncode: int = 0, output: str = "") -> MagicMock:
    result = MagicMock()
    result.data = {"returncode": returncode, "output": output}
    return result


def _make_sandbox(exec_result=None, file_content: bytes = b"audio_data") -> AsyncMock:
    sandbox = AsyncMock()
    sandbox.exec_command = AsyncMock(
        return_value=exec_result or _make_exec_result(0, "")
    )
    sandbox.write_file = AsyncMock()
    sandbox.download_file = AsyncMock(return_value=io.BytesIO(file_content))
    return sandbox


def _make_whisper_output(language: str = "zh", segments=None) -> str:
    if segments is None:
        segments = [
            {"start": 0.0, "end": 2.5, "text": " Hello world"},
            {"start": 2.5, "end": 5.0, "text": " Goodbye"},
        ]
    return json.dumps({"language": language, "segments": segments}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAudioFileProcessorDisabled:
    def test_disabled_provider_returns_info_message(self):
        config = _make_config(provider="disabled")
        proc = AudioFileProcessor(sandbox=_make_sandbox(), config=config)
        result = asyncio.run(
            proc.process("/tmp/audio.mp3", "audio.mp3", "audio/mpeg", supports_vision=False)
        )
        assert isinstance(result, FileProcessResult)
        assert "audio.mp3" in result.text
        assert "not configured" in result.text
        assert result.image_blocks == ()


class TestAudioFileProcessorUnknownProvider:
    def test_unknown_provider_returns_error_message(self):
        config = _make_config(provider="unknown_engine")
        proc = AudioFileProcessor(sandbox=_make_sandbox(), config=config)
        result = asyncio.run(
            proc.process("/tmp/audio.mp3", "audio.mp3", "audio/mpeg", supports_vision=False)
        )
        assert "unknown provider" in result.text
        assert "unknown_engine" in result.text


class TestAudioFileProcessorSandboxWhisper:
    def test_sandbox_whisper_returns_formatted_transcript(self):
        whisper_out = _make_whisper_output()
        ffprobe_result = _make_exec_result(returncode=0, output="125.0\n")
        preprocess_result = _make_exec_result(returncode=0, output="")
        whisper_result = _make_exec_result(returncode=0, output=whisper_out)
        cleanup_result = _make_exec_result(returncode=0, output="")

        sandbox = _make_sandbox()
        sandbox.exec_command = AsyncMock(side_effect=[ffprobe_result, preprocess_result, whisper_result, cleanup_result])

        config = _make_config(provider="sandbox_whisper")
        proc = AudioFileProcessor(sandbox=sandbox, config=config)
        result = asyncio.run(
            proc.process("/tmp/audio.mp3", "audio.mp3", "audio/mpeg", supports_vision=False)
        )

        assert isinstance(result, FileProcessResult)
        assert "audio.mp3" in result.text
        # duration: 125s = 2m05s
        assert "02:05" in result.text
        # transcript content
        assert "Hello world" in result.text
        assert "Language: zh" in result.text

    def test_sandbox_whisper_exec_failure_returns_error(self):
        ffprobe_result = _make_exec_result(returncode=1, output="")
        preprocess_result = _make_exec_result(returncode=0, output="")
        whisper_result = _make_exec_result(returncode=1, output="ModuleNotFoundError: No module named 'faster_whisper'")
        cleanup_result = _make_exec_result(returncode=0, output="")

        sandbox = _make_sandbox()
        sandbox.exec_command = AsyncMock(side_effect=[ffprobe_result, preprocess_result, whisper_result, cleanup_result])

        config = _make_config(provider="sandbox_whisper")
        proc = AudioFileProcessor(sandbox=sandbox, config=config)
        result = asyncio.run(
            proc.process("/tmp/audio.mp3", "audio.mp3", "audio/mpeg", supports_vision=False)
        )

        assert "Transcription failed" in result.text

    def test_sandbox_whisper_writes_script_to_tmp(self):
        whisper_out = _make_whisper_output()
        ffprobe_result = _make_exec_result(0, "")
        preprocess_result = _make_exec_result(0, "")
        whisper_result = _make_exec_result(0, whisper_out)
        cleanup_result = _make_exec_result(0, "")

        sandbox = _make_sandbox()
        sandbox.exec_command = AsyncMock(side_effect=[ffprobe_result, preprocess_result, whisper_result, cleanup_result])

        config = _make_config(provider="sandbox_whisper")
        proc = AudioFileProcessor(sandbox=sandbox, config=config)
        asyncio.run(
            proc.process("/tmp/audio.mp3", "audio.mp3", "audio/mpeg", supports_vision=False)
        )

        sandbox.write_file.assert_awaited_once()
        call_args = sandbox.write_file.call_args
        assert call_args[0][0] == "/tmp/_audio_transcribe.py"
        assert "faster_whisper" in call_args[0][1]


class TestAudioFileProcessorOpenAI:
    def test_openai_api_path_returns_formatted_transcript(self):
        segments = [
            {"start": 0.0, "end": 3.0, "text": " Testing OpenAI transcription"},
        ]
        openai_response = {
            "text": "Testing OpenAI transcription",
            "segments": segments,
            "language": "en",
        }

        ffprobe_result = _make_exec_result(0, "63.0\n")
        sandbox = _make_sandbox()
        sandbox.exec_command = AsyncMock(return_value=ffprobe_result)
        sandbox.download_file = AsyncMock(return_value=io.BytesIO(b"fake_audio"))

        config = _make_config(provider="openai_api")
        proc = AudioFileProcessor(sandbox=sandbox, config=config)

        mock_resp = MagicMock()
        mock_resp.json.return_value = openai_response
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(
                proc.process("/tmp/audio.mp3", "audio.mp3", "audio/mpeg", supports_vision=False)
            )

        assert isinstance(result, FileProcessResult)
        assert "audio.mp3" in result.text
        assert "Testing OpenAI transcription" in result.text
        # duration: 63s = 1m03s
        assert "01:03" in result.text

    def test_openai_oversized_returns_error(self):
        big_audio = b"x" * (101 * 1024 * 1024)
        ffprobe_result = _make_exec_result(0, "")
        sandbox = _make_sandbox(file_content=big_audio)
        sandbox.exec_command = AsyncMock(return_value=ffprobe_result)

        config = _make_config(provider="openai_api")
        proc = AudioFileProcessor(sandbox=sandbox, config=config)

        result = asyncio.run(
            proc.process("/tmp/big.mp3", "big.mp3", "audio/mpeg", supports_vision=False)
        )

        assert "too large" in result.text


class TestFormatSegments:
    def test_empty_segments_returns_fallback_text(self):
        result = AudioFileProcessor._format_segments([], fallback_text="No speech")
        assert result == "No speech"

    def test_empty_segments_no_fallback_returns_default_message(self):
        result = AudioFileProcessor._format_segments([])
        assert result == "[No speech detected]"

    def test_valid_segments_formatted_with_timestamps(self):
        segments = [
            {"start": 0.0, "end": 5.5, "text": " Hello"},
            {"start": 5.5, "end": 12.0, "text": " World"},
        ]
        result = AudioFileProcessor._format_segments(segments)
        assert "[00:00 - 00:05] Hello" in result
        assert "[00:05 - 00:12] World" in result

    def test_language_hint_prepended(self):
        segments = [{"start": 0.0, "end": 1.0, "text": " Test"}]
        result = AudioFileProcessor._format_segments(segments, language_hint="en")
        lines = result.split("\n")
        assert lines[0] == "Language: en"
        assert lines[1] == ""

    def test_segments_over_60_seconds_formatted_correctly(self):
        segments = [
            {"start": 65.0, "end": 125.0, "text": " Long segment"},
        ]
        result = AudioFileProcessor._format_segments(segments)
        assert "[01:05 - 02:05] Long segment" in result


class TestTruncateTranscript:
    def test_short_transcript_not_truncated(self):
        text = "Header\n\nBody text that is short"
        result = AudioFileProcessor._truncate_transcript(text, max_chars=1000)
        assert result == text

    def test_long_transcript_gets_truncated(self):
        header = "[Audio: file.mp3, 10:00]\n\nLanguage: en"
        body = "\n".join(
            [f"[{i:02d}:{i:02d} - {i:02d}:{i+1:02d}] Segment {i}" for i in range(200)]
        )
        full_text = header + "\n" + body
        result = AudioFileProcessor._truncate_transcript(full_text, max_chars=500)
        assert len(result) <= 550  # allow small margin for the truncation marker
        assert "...(已省略部分内容)" in result


class TestDurationFormatting:
    def _run_get_duration(self, output: str, returncode: int = 0) -> str | None:
        exec_result = _make_exec_result(returncode, output)
        sandbox = _make_sandbox(exec_result=exec_result)
        config = _make_config()
        proc = AudioFileProcessor(sandbox=sandbox, config=config)
        return asyncio.run(
            proc._get_duration("/tmp/audio.mp3")
        )

    def test_duration_minutes_seconds(self):
        result = self._run_get_duration("125.0\n")
        assert result == "02:05"

    def test_duration_hours_minutes_seconds(self):
        result = self._run_get_duration("3723.0\n")  # 1h 2m 3s
        assert result == "1:02:03"

    def test_duration_ffprobe_failure_returns_none(self):
        result = self._run_get_duration("", returncode=1)
        assert result is None

    def test_duration_ffprobe_invalid_output_returns_none(self):
        exec_result = _make_exec_result(0, "not_a_number\n")
        sandbox = _make_sandbox(exec_result=exec_result)
        config = _make_config()
        proc = AudioFileProcessor(sandbox=sandbox, config=config)
        result = asyncio.run(
            proc._get_duration("/tmp/audio.mp3")
        )
        assert result is None


class TestRegistryAudioIntegration:
    def test_audio_enabled_registers_processor(self):
        from app.infrastructure.external.file_processors.registry import FileProcessorRegistry

        audio_config = AudioProcessorConfig(provider="sandbox_whisper")
        registry = FileProcessorRegistry(
            sandbox=AsyncMock(),
            file_uploader=AsyncMock(),
            audio_config=audio_config,
        )
        proc = registry.get_processor("audio/mpeg")
        assert proc is not None
        assert type(proc).__name__ == "AudioFileProcessor"

    def test_audio_stored_as_audio_processor_attribute(self):
        from app.infrastructure.external.file_processors.registry import FileProcessorRegistry

        audio_config = AudioProcessorConfig(provider="openai_api", openai_api_key="sk-x")
        registry = FileProcessorRegistry(
            sandbox=AsyncMock(),
            file_uploader=AsyncMock(),
            audio_config=audio_config,
        )
        assert registry._audio_processor is not None
        assert type(registry._audio_processor).__name__ == "AudioFileProcessor"

    def test_audio_disabled_audio_processor_is_none(self):
        from app.infrastructure.external.file_processors.registry import FileProcessorRegistry

        registry = FileProcessorRegistry(
            sandbox=AsyncMock(),
            file_uploader=AsyncMock(),
        )
        assert registry._audio_processor is None
        assert registry.get_processor("audio/mpeg") is None

    def test_audio_wav_matches_audio_prefix(self):
        from app.infrastructure.external.file_processors.registry import FileProcessorRegistry

        audio_config = AudioProcessorConfig(provider="sandbox_whisper")
        registry = FileProcessorRegistry(
            sandbox=AsyncMock(),
            file_uploader=AsyncMock(),
            audio_config=audio_config,
        )
        proc = registry.get_processor("audio/wav")
        assert proc is not None
