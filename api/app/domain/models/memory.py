import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_BROWSER_COMPACT_TOOLS = frozenset(["browser_view", "browser_navigate"])
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


def _extract_browser_summary(function_name: str, content: str) -> str:
    """从 browser 工具的 HTML 内容中提取摘要。"""
    title_match = _TITLE_RE.search(content)
    title = title_match.group(1).strip() if title_match else ""
    plain_text = re.sub(r"\s+", " ", _TAG_RE.sub(" ", content)).strip()
    text_preview = plain_text[:200]

    parts = [f"[已执行] {function_name}:"]
    if title:
        parts.append(f"页面标题: {title},")
    parts.append(f"内容摘要: {text_preview}")
    return " ".join(parts)


class Memory(BaseModel):
    """记忆类，定义Agent的记忆基础信息"""

    messages: List[Dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def get_message_role(cls, message: Dict[str, Any]) -> str:
        """根据传递的消息来获取消息的角色信息"""
        return message.get("role")

    def add_message(self, message: Dict[str, Any]) -> None:
        """往记忆中添加一条消息"""
        self.messages.append(message)

    def add_messages(self, messages: List[Dict[str, Any]]) -> None:
        """往记忆中添加多条消息"""
        self.messages.extend(messages)

    def get_messages(self) -> List[Dict[str, Any]]:
        """获取记忆中的所有消息列表"""
        return self.messages

    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """获取记忆中的最后一条消息，如果不存在则返回None"""
        return self.messages[-1] if len(self.messages) > 0 else None

    def roll_back(self) -> None:
        """回滚记忆，删除最后一条消息"""
        self.messages = self.messages[:-1]

    def compact(self, keep_summary: bool = False) -> None:
        """记忆压缩，将记忆中已经执行的工具(搜索/网页源码获取/浏览器访问结果等)这类已经执行过的消息进行压缩检索"""
        # 1.循环遍历所有的消息列表
        for message in self.messages:
            # 2.判断消息的角色是否为tool
            if self.get_message_role(message) == "tool":
                if message.get("function_name") in _BROWSER_COMPACT_TOOLS:
                    if keep_summary:
                        message["content"] = _extract_browser_summary(
                            message["function_name"], message.get("content", "")
                        )
                    else:
                        message["content"] = "(removed)"
                    logger.debug(
                        f"从记忆中压缩对应工具的结果: {message['function_name']}"
                    )
            # 3.压缩记忆时reasoning_content内容可以去除压缩上下文
            if "reasoning_content" in message:
                logger.debug(
                    f"从记忆中移除工具思考结果: {message['reasoning_content'][:50]}..."
                )
                del message["reasoning_content"]

    @property
    def empty(self) -> bool:
        """只读属性，检查记忆是否为空"""
        return len(self.messages) == 0
