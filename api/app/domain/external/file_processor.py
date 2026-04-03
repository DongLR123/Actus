from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


# Shared constant: max image blocks per file_view call.
# Used by react_graph (tool_node truncation) and video processor (self-limiting).
MAX_FILE_VIEW_IMAGES = 10


@dataclass(frozen=True)
class FileProcessResult:
    """文件处理结果。text 进入 ToolMessage，image_blocks 注入 HumanMessage。"""
    text: str
    image_blocks: tuple[dict, ...] = ()
    document_blocks: tuple[dict, ...] = ()


class FileProcessor(Protocol):
    """文件处理器协议。每种文件类型一个实现。"""

    async def process(
        self,
        sandbox_path: str,
        filename: str,
        mime_type: str,
        supports_vision: bool,
        supports_pdf_input: bool = False,
    ) -> FileProcessResult: ...


class FileProcessorLookup(Protocol):
    """文件处理器查找协议。domain 层使用此协议。"""

    def get_processor(self, mime_type: str) -> FileProcessor | None: ...
