from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from app.domain.external.file_processor import FileProcessor
from app.domain.external.sandbox import Sandbox
from app.infrastructure.external.file_processors.image import ImageFileProcessor

logger = logging.getLogger(__name__)

FileUploader = Callable[[bytes, str], Awaitable[str | None]]


class FileProcessorRegistry:
    """MIME 前缀 → Processor 映射。实现 domain FileProcessorLookup Protocol。"""

    def __init__(
        self,
        sandbox: Sandbox,
        file_uploader: FileUploader,
        vision_model: Any | None = None,
        audio_config: Any | None = None,
        video_config: Any | None = None,
    ) -> None:
        self._processors: list[tuple[str, FileProcessor]] = []

        self._processors.append((
            "image/",
            ImageFileProcessor(
                sandbox=sandbox,
                file_uploader=file_uploader,
                vision_model=vision_model,
            ),
        ))

        try:
            from app.infrastructure.external.file_processors.pdf import PdfFileProcessor
            self._processors.append((
                "application/pdf",
                PdfFileProcessor(sandbox=sandbox, file_uploader=file_uploader),
            ))
        except ImportError:
            logger.debug("PdfFileProcessor not available, skipping PDF support")

        if audio_config and getattr(audio_config, "provider", "disabled") != "disabled":
            try:
                from app.infrastructure.external.file_processors.audio import AudioFileProcessor
                self._processors.append((
                    "audio/",
                    AudioFileProcessor(sandbox=sandbox, config=audio_config),
                ))
            except ImportError:
                logger.debug("AudioFileProcessor not available, skipping audio support")

        if video_config:
            try:
                from app.infrastructure.external.file_processors.video import VideoFileProcessor
                self._processors.append((
                    "video/",
                    VideoFileProcessor(
                        sandbox=sandbox,
                        file_uploader=file_uploader,
                        audio_config=audio_config,
                        video_config=video_config,
                    ),
                ))
            except ImportError:
                logger.debug("VideoFileProcessor not available, skipping video support")

    # MIME types that match a prefix but have no working processor
    _EXCLUDED_MIMES = frozenset({"image/svg+xml"})

    def get_processor(self, mime_type: str) -> FileProcessor | None:
        if mime_type in self._EXCLUDED_MIMES:
            return None
        for prefix, processor in self._processors:
            if mime_type.startswith(prefix) or mime_type == prefix:
                return processor
        return None
