"""Audio file processor — dual engine: OpenAI API or sandbox faster-whisper."""
from __future__ import annotations

import asyncio
import json
import logging
import shlex
import uuid

from app.domain.external.file_processor import FileProcessResult
from app.domain.external.sandbox import Sandbox
from app.domain.models.app_config import AudioProcessorConfig

logger = logging.getLogger(__name__)

_MAX_AUDIO_SIZE = 100 * 1024 * 1024  # 100MB
_MAX_TRANSCRIPT_CHARS = 8000

_WHISPER_SCRIPT = """from faster_whisper import WhisperModel
import json, sys
model = WhisperModel("base", compute_type="int8")
segments, info = model.transcribe(sys.argv[1])
output = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
print(json.dumps({"language": info.language, "segments": output}, ensure_ascii=False))"""


class AudioFileProcessor:
    def __init__(self, sandbox: Sandbox, config: AudioProcessorConfig) -> None:
        self._sandbox = sandbox
        self._config = config

    async def process(
        self,
        sandbox_path: str,
        filename: str,
        mime_type: str,
        supports_vision: bool,
        supports_pdf_input: bool = False,
    ) -> FileProcessResult:
        if self._config.provider == "disabled":
            return FileProcessResult(
                text=f"[Audio: {filename} — audio processing not configured]"
            )

        # Size preflight (reliable: download + measure, shared across all providers)
        file_io = await self._sandbox.download_file(sandbox_path)
        audio_bytes = file_io.read() if hasattr(file_io, "read") else file_io
        if len(audio_bytes) > _MAX_AUDIO_SIZE:
            return FileProcessResult(
                text=f"[Audio: {filename}, {len(audio_bytes)} bytes — too large (limit {_MAX_AUDIO_SIZE // (1024*1024)}MB)]"
            )
        # Cache bytes for openai_api path (avoids double download)
        self._cached_bytes: tuple[str, bytes] = (sandbox_path, audio_bytes)

        # Get duration via ffprobe
        duration_str = await self._get_duration(sandbox_path)

        if self._config.provider == "openai_api":
            transcript = await self._transcribe_openai(sandbox_path, filename)
        elif self._config.provider == "sandbox_whisper":
            transcript = await self._transcribe_sandbox(sandbox_path)
        else:
            return FileProcessResult(
                text=f"[Audio: {filename} — unknown provider: {self._config.provider}]"
            )

        header = f"[Audio: {filename}"
        if duration_str:
            header += f", {duration_str}"
        header += "]"

        full_text = f"{header}\n\n{transcript}"
        if len(full_text) > _MAX_TRANSCRIPT_CHARS:
            full_text = self._truncate_transcript(full_text, _MAX_TRANSCRIPT_CHARS)

        return FileProcessResult(text=full_text)

    async def _get_duration(self, sandbox_path: str) -> str | None:
        try:
            result = await asyncio.wait_for(
                self._sandbox.exec_command(
                    "default",
                    "",
                    f"ffprobe -v quiet -show_entries format=duration"
                    f" -of csv=p=0 {shlex.quote(sandbox_path)}",
                ),
                timeout=10.0,
            )
            if hasattr(result, "data") and isinstance(result.data, dict):
                if result.data.get("returncode", -1) == 0:
                    raw = result.data.get("output", "").strip()
                    secs = float(raw)
                    mins, secs_r = divmod(int(secs), 60)
                    hrs, mins_r = divmod(mins, 60)
                    if hrs > 0:
                        return f"{hrs}:{mins_r:02d}:{secs_r:02d}"
                    return f"{mins_r:02d}:{secs_r:02d}"
        except Exception:
            pass
        return None

    async def _transcribe_openai(self, sandbox_path: str, filename: str) -> str:
        import httpx

        # Use cached bytes from process() preflight to avoid double download
        if hasattr(self, "_cached_bytes") and self._cached_bytes[0] == sandbox_path:
            audio_bytes = self._cached_bytes[1]
        else:
            file_io = await self._sandbox.download_file(sandbox_path)
            audio_bytes = file_io.read() if hasattr(file_io, "read") else file_io

        base_url = self._config.openai_base_url.rstrip("/")
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url}/audio/transcriptions",
                headers={"Authorization": f"Bearer {self._config.openai_api_key}"},
                files={"file": (filename, audio_bytes)},
                data={
                    "model": self._config.openai_model,
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "segment",
                },
                timeout=300.0,
            )
            resp.raise_for_status()
            result = resp.json()
        return self._format_segments(
            result.get("segments", []), result.get("text", "")
        )

    async def _transcribe_sandbox(self, sandbox_path: str) -> str:
        # Step 1: ffmpeg preprocessing — convert to WAV 16kHz mono for faster-whisper
        wav_path = f"/tmp/_audio_{uuid.uuid4().hex[:8]}.wav"
        preprocess_cmd = (
            f"ffmpeg -i {shlex.quote(sandbox_path)} "
            f"-vn -acodec pcm_s16le -ar 16000 -ac 1 "
            f"{shlex.quote(wav_path)} -y 2>/dev/null"
        )
        preprocess_result = await asyncio.wait_for(
            self._sandbox.exec_command("default", "", preprocess_cmd),
            timeout=60.0,
        )
        # Use preprocessed WAV if conversion succeeded, otherwise try original file
        transcribe_path = sandbox_path
        if hasattr(preprocess_result, "data") and isinstance(preprocess_result.data, dict):
            if preprocess_result.data.get("returncode", -1) == 0:
                transcribe_path = wav_path
            else:
                logger.warning("Audio preprocessing failed, using original file")

        # Step 2: Run faster-whisper transcription
        script_path = "/tmp/_audio_transcribe.py"
        await self._sandbox.write_file(script_path, _WHISPER_SCRIPT)
        result = await asyncio.wait_for(
            self._sandbox.exec_command(
                "default",
                "",
                f"python3 {shlex.quote(script_path)} {shlex.quote(transcribe_path)}",
            ),
            timeout=300.0,
        )

        # Cleanup preprocessed file
        try:
            await self._sandbox.exec_command("default", "", f"rm -f {shlex.quote(wav_path)}")
        except Exception:
            pass

        if hasattr(result, "data") and isinstance(result.data, dict):
            if result.data.get("returncode", -1) != 0:
                error = result.data.get("output", "")[:500]
                return f"[Transcription failed: {error}]"
            output = result.data.get("output", "")
        else:
            output = str(result)

        data = json.loads(output)
        language = data.get("language", "unknown")
        return self._format_segments(data.get("segments", []), language_hint=language)

    @staticmethod
    def _format_segments(
        segments: list[dict],
        fallback_text: str = "",
        language_hint: str = "",
    ) -> str:
        if not segments:
            return fallback_text or "[No speech detected]"

        lines: list[str] = []
        if language_hint:
            lines.append(f"Language: {language_hint}")
            lines.append("")
        for seg in segments:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = seg.get("text", "").strip()
            s_min, s_sec = divmod(int(start), 60)
            e_min, e_sec = divmod(int(end), 60)
            lines.append(f"[{s_min:02d}:{s_sec:02d} - {e_min:02d}:{e_sec:02d}] {text}")
        return "\n".join(lines)

    @staticmethod
    def _truncate_transcript(text: str, max_chars: int) -> str:
        lines = text.split("\n")
        # Keep header (first 2 lines) + beginning + end
        header = "\n".join(lines[:3])
        body = "\n".join(lines[3:])
        if len(header) + len(body) <= max_chars:
            return text
        budget = max_chars - len(header) - 50  # margin for truncation marker
        half = budget // 2
        if half <= 0:
            return text[:max_chars]
        return (
            header
            + "\n"
            + body[:half]
            + "\n...(已省略部分内容)\n"
            + body[-half:]
        )
