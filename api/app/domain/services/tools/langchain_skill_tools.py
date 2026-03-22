"""LangChain tool wrappers for Skill creation and discovery tools.

Wraps BrainstormSkillTool and CreateSkillTool (legacy BaseTool instances)
as LangChain StructuredTool functions so they can be used in the LangGraph
react_graph.

Also provides ``create_skill_guide_tool`` for on-demand SKILL.md loading,
following the Claude Code two-phase pattern: metadata discovery first,
full guide on invocation.

Usage:
    tools = create_skill_langchain_tools(
        brainstorm_skill_tool=brainstorm_tool,
        create_skill_tool=create_tool,
    )
    guide_tool = create_skill_guide_tool(skill_pool_ref)
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Callable, Optional

from langchain_core.tools import StructuredTool, tool as lc_tool

from app.domain.services.tools.base import BaseTool

if TYPE_CHECKING:
    from app.domain.models.skill import Skill

logger = logging.getLogger(__name__)


def create_skill_langchain_tools(
    brainstorm_skill_tool: BaseTool | None = None,
    create_skill_tool: BaseTool | None = None,
) -> list[StructuredTool]:
    """Create LangChain wrappers for skill creation tools.

    Returns an empty list if the corresponding BaseTool instance is None.
    """
    tools: list[StructuredTool] = []

    if brainstorm_skill_tool is not None:

        @lc_tool
        async def brainstorm_skill(description: str) -> str:
            """根据需求描述生成 Skill 蓝图预览（名称、工具列表、参数、依赖），供用户确认后再正式创建。"""
            result = await brainstorm_skill_tool.invoke(
                "brainstorm_skill", description=description,
            )
            return result.model_dump_json()

        tools.append(brainstorm_skill)

    if create_skill_tool is not None:

        @lc_tool
        async def generate_skill(
            description: str,
            blueprint: Optional[dict] = None,
            blueprint_json: Optional[str] = "",
        ) -> str:
            """生成 Skill 代码并在沙箱验证。返回生成结果和验证状态，不自动安装。用户确认后再调用 install_skill 完成安装。"""
            kwargs: dict = {"description": description}
            if blueprint is not None:
                kwargs["blueprint"] = blueprint
            if blueprint_json:
                kwargs["blueprint_json"] = blueprint_json
            result = await create_skill_tool.invoke("generate_skill", **kwargs)
            return result.model_dump_json()

        @lc_tool
        async def install_skill(skill_data: str) -> str:
            """安装已生成并验证通过的 Skill。传入 generate_skill 返回的 data.skill_data JSON 字符串。"""
            result = await create_skill_tool.invoke(
                "install_skill", skill_data=skill_data,
            )
            return result.model_dump_json()

        tools.append(generate_skill)
        tools.append(install_skill)

    return tools


# ---------------------------------------------------------------------------
# On-demand Skill guide loading (Claude Code two-phase pattern)
# ---------------------------------------------------------------------------

_SKILL_GUIDE_MAX_CHARS = 4000


def _strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter from SKILL.md content."""
    raw = (text or "").strip()
    if not raw.startswith("---"):
        return raw
    lines = raw.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return raw
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            return "\n".join(lines[idx + 1:]).strip()
    return raw


def _extract_guide_body(skill: "Skill") -> str:
    """Extract full guide content from a Skill's manifest."""
    manifest = skill.manifest if isinstance(skill.manifest, dict) else {}
    context_blob = str(manifest.get("context_blob") or "").strip()
    if context_blob:
        body = context_blob
    else:
        skill_md = str(manifest.get("skill_md") or "").strip()
        body = _strip_frontmatter(skill_md)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    if len(body) > _SKILL_GUIDE_MAX_CHARS:
        body = body[:_SKILL_GUIDE_MAX_CHARS].rstrip() + "\n...(truncated)"
    return body or (skill.description or "").strip() or "No guide content available."


def create_skill_guide_tool(
    skill_pool_ref: Callable[[], list["Skill"]],
) -> StructuredTool:
    """Create a ``get_skill_guide`` tool for on-demand SKILL.md loading.

    Follows the **Claude Code two-phase pattern**:
    - Phase 1 (discovery): Skill names + descriptions shown in Active Skills context
    - Phase 2 (loading): LLM calls this tool to get the full SKILL.md guide

    Parameters
    ----------
    skill_pool_ref :
        A callable that returns the current session skill pool.
        Using a callable (rather than a direct list) ensures the tool
        always reads the latest pool state.
    """

    async def _get_skill_guide(skill_slug: str) -> str:
        pool = skill_pool_ref()
        # Match by slug (primary) or name (fallback)
        slug_lower = skill_slug.strip().lower()
        for skill in pool:
            if (skill.slug or "").lower() == slug_lower:
                guide = _extract_guide_body(skill)
                logger.info("[SkillGuide] Loaded guide for '%s' (%d chars)", skill.slug, len(guide))
                return f"# {skill.name}\n\n{guide}"
        # Fallback: match by name
        for skill in pool:
            if (skill.name or "").lower() == slug_lower:
                guide = _extract_guide_body(skill)
                logger.info("[SkillGuide] Loaded guide for '%s' (by name, %d chars)", skill.name, len(guide))
                return f"# {skill.name}\n\n{guide}"
        available = [s.slug for s in pool if s.slug]
        available_str = ", ".join(available) if available else "(none)"
        return f"Skill '{skill_slug}' not found. Available skills: {available_str}"

    return StructuredTool.from_function(
        coroutine=_get_skill_guide,
        name="get_skill_guide",
        description=(
            "获取指定 Skill 的完整使用指南（SKILL.md）。"
            "当 Active Skills 中的简短描述不够用时，调用此工具获取详细的操作步骤、代码示例和最佳实践。"
            "传入 skill 的 slug（括号中的标识符，如 'xlsx'、'frontend-design'）。"
        ),
    )
