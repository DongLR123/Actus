from __future__ import annotations

import asyncio
import io
import logging
from typing import Any, Awaitable, Callable

from app.domain.external.file_processor import FileProcessResult
from app.domain.external.sandbox import Sandbox

logger = logging.getLogger(__name__)

FileUploader = Callable[[bytes, str], Awaitable[str | None]]

_MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB


class ImageFileProcessor:
    """图片文件处理器。

    supports_vision=True: 上传 MinIO → presigned URL → image_url block
    supports_vision=False + vision_model: 调视觉模型描述 → 纯文本
    supports_vision=False 无 vision_model: 仅返回元数据
    """

    def __init__(
        self,
        sandbox: Sandbox,
        file_uploader: FileUploader,
        vision_model: Any | None = None,
    ) -> None:
        self._sandbox = sandbox
        self._file_uploader = file_uploader
        self._vision_model = vision_model

    async def process(
        self,
        sandbox_path: str,
        filename: str,
        mime_type: str,
        supports_vision: bool,
    ) -> FileProcessResult:
        file_io = await self._sandbox.download_file(sandbox_path)
        file_bytes = file_io.read() if hasattr(file_io, "read") else file_io

        if len(file_bytes) > _MAX_IMAGE_SIZE:
            return FileProcessResult(
                text=f"[Image: {filename}, {len(file_bytes)} bytes — too large to process]"
            )

        w, h = await asyncio.to_thread(self._get_dimensions, file_bytes)
        url = await self._file_uploader(file_bytes, filename)
        text = f"[Image: {filename}, {w}x{h}]"

        if supports_vision and url:
            return FileProcessResult(
                text=text,
                image_blocks=(
                    {"type": "image_url", "image_url": {"url": url, "detail": "auto"}},
                ),
            )
        elif not supports_vision and self._vision_model and url:
            desc = await self._describe_with_vision(url)
            return FileProcessResult(text=f"{text}\n{desc}")
        else:
            return FileProcessResult(text=text)

    @staticmethod
    def _get_dimensions(file_bytes: bytes) -> tuple[int, int]:
        from PIL import Image

        img = Image.open(io.BytesIO(file_bytes))
        return img.size

    async def _describe_with_vision(self, image_url: str) -> str:
        from langchain_core.messages import HumanMessage

        msg = HumanMessage(content=[
            {"type": "text", "text": "Describe this image concisely. Focus on key visual elements, text, layout, and purpose."},
            {"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}},
        ])
        try:
            response = await asyncio.wait_for(
                self._vision_model.ainvoke([msg]),
                timeout=30.0,
            )
            return response.content if isinstance(response.content, str) else str(response.content)
        except Exception as e:
            logger.warning("Vision fallback failed for %s: %s", image_url, e)
            return "[Vision analysis unavailable]"
