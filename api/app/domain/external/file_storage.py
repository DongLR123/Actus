from typing import BinaryIO, Protocol, Tuple

from app.domain.models.file import File
from fastapi import UploadFile


class FileStorage(Protocol):
    """文件存储桶协议"""

    async def upload_file(self, upload_file: UploadFile) -> File:
        """根据传递的文件源上传文件后返回文件信息"""
        ...

    async def download_file(self, file_id: str) -> Tuple[BinaryIO, File]:
        """根据传递的文件id下载文件，并返回文件源+文件信息"""
        ...

    async def delete_file(self, file_id: str) -> None:
        """根据传递的文件id删除文件"""
        ...

    async def get_presigned_url(
        self, file: File, expiry_seconds: int = 86400
    ) -> str | None:
        """生成文件的预签名访问 URL。不支持时返回 None。"""
        return None
