from typing import List, Literal

from pydantic import BaseModel, Field

SkillConfirmationAction = Literal[
    "generate", "revise", "install", "cancel", "regenerate", "retry"
]


class Message(BaseModel):
    """用户传递的消息"""

    message: str = ""  # 用户发送的消息
    attachments: List[str] = Field(default_factory=list)  # 用户发送的附件
    image_content_blocks: List[dict] = Field(default_factory=list)  # 图片附件的多模态内容块
    skill_confirmation_action: SkillConfirmationAction | None = None
