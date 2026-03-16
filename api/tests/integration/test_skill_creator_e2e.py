"""Skill Creator 端到端（mock）测试。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.skill_creator_service import SkillCreatorService
from app.domain.models.skill_creator import (
    ScriptFile,
    SkillBlueprint,
    SkillCreationProgress,
    SkillCreationResult,
    SkillGeneratedFiles,
    ToolDef,
)
from app.domain.models.tool_result import ToolResult

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_mock_llm(
    structured_returns: list | None = None,
    single_return=None,
    direct_returns: list | None = None,
    direct_return=None,
):
    """Build a mock BaseChatModel that supports both call patterns."""
    import json as _json

    llm = MagicMock()
    structured_runnable = AsyncMock()

    if structured_returns is not None:
        structured_runnable.ainvoke = AsyncMock(side_effect=structured_returns)
    elif single_return is not None:
        structured_runnable.ainvoke = AsyncMock(return_value=single_return)
    else:
        structured_runnable.ainvoke = AsyncMock()

    llm.with_structured_output = MagicMock(return_value=structured_runnable)

    def _to_ai_message(val):
        if hasattr(val, "model_dump_json"):
            return SimpleNamespace(content=val.model_dump_json())
        if isinstance(val, dict):
            return SimpleNamespace(content=_json.dumps(val, ensure_ascii=False))
        if isinstance(val, str):
            return SimpleNamespace(content=val)
        return val

    if direct_returns is not None:
        llm.ainvoke = AsyncMock(side_effect=[_to_ai_message(v) for v in direct_returns])
    elif direct_return is not None:
        llm.ainvoke = AsyncMock(return_value=_to_ai_message(direct_return))
    else:
        llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content="{}"))

    return llm, structured_runnable


async def test_full_pipeline_with_mocked_llm_and_sandbox() -> None:
    mock_github = MagicMock()
    mock_github.research_keywords = AsyncMock(return_value=[])
    mock_github.format_research_report = MagicMock(return_value="无参考")
    mock_skill_service = AsyncMock()

    mock_skill_service.install_skill = AsyncMock(
        return_value=SimpleNamespace(id="echo--abc12345", name="echo")
    )

    mock_sandbox = AsyncMock()
    mock_sandbox.write_file = AsyncMock(return_value=ToolResult(success=True, message="ok"))
    mock_sandbox.exec_command = AsyncMock(return_value=ToolResult(success=True, message='{"result": "ok"}'))

    analyze_blueprint = SkillBlueprint(
        skill_name="echo",
        description="Echo back input",
        tools=[
            ToolDef(
                name="echo",
                description="Echo",
                parameters=[],
            )
        ],
        search_keywords=["echo python"],
        estimated_deps=[],
    )
    generate_files = SkillGeneratedFiles(
        skill_md="---\nname: echo\nruntime_type: native\n---\n# Echo",
        manifest={
            "name": "echo",
            "slug": "echo",
            "version": "0.1.0",
            "description": "Echo",
            "runtime_type": "native",
            "tools": [
                {
                    "name": "echo",
                    "description": "Echo",
                    "parameters": {"text": {"type": "string"}},
                    "required": ["text"],
                    "entry": {"command": "python bundle/echo.py"},
                }
            ],
        },
        scripts=[
            ScriptFile(
                path="bundle/echo.py",
                content="import sys, json\nargs = json.loads(sys.argv[1]) if len(sys.argv)>1 else {}\nprint(json.dumps({'result': args.get('text', '')}))",
            )
        ],
        dependencies=[],
    )
    mock_llm, _ = _make_mock_llm(
        structured_returns=[analyze_blueprint, generate_files],
    )

    service = SkillCreatorService(
        llm=mock_llm,
        github_client=mock_github,
        skill_service=mock_skill_service,
    )

    events = []
    async for event in service.create(
        description="创建一个 echo skill",
        sandbox=mock_sandbox,
        installed_by="test-user",
    ):
        events.append(event)

    progress_events = [event for event in events if isinstance(event, SkillCreationProgress)]
    result_events = [event for event in events if isinstance(event, SkillCreationResult)]

    assert len(result_events) == 1
    assert result_events[0].skill_name == "echo"
    assert "echo" in result_events[0].tools

    steps = [event.step for event in progress_events]
    assert "analyzing" in steps
    assert "researching" in steps
    assert "generating" in steps
    assert "validating" in steps
    assert "installing" in steps

    mock_skill_service.install_skill.assert_called_once()


async def test_brainstorm_generate_install_flow() -> None:
    """端到端测试：brainstorm (analyze) → generate → install 完整流程。"""
    mock_github = MagicMock()
    mock_github.research_keywords = AsyncMock(return_value=[])
    mock_github.format_research_report = MagicMock(return_value="无参考")
    mock_skill_service = AsyncMock()
    mock_skill_service.install_skill = AsyncMock(
        return_value=SimpleNamespace(id="translator--abc12345", name="translator")
    )
    mock_sandbox = AsyncMock()
    mock_sandbox.write_file = AsyncMock(return_value=ToolResult(success=True, message="ok"))
    mock_sandbox.exec_command = AsyncMock(return_value=ToolResult(success=True, message='{"result": "ok"}'))

    # Step 1: brainstorm (analyze)
    analyze_blueprint = SkillBlueprint(
        skill_name="translator",
        description="中英翻译",
        tools=[ToolDef(name="translate", description="翻译", parameters=[])],
        search_keywords=["translate python"],
        estimated_deps=["googletrans"],
    )

    # Step 2: generate
    generated = SkillGeneratedFiles(
        skill_md="---\nname: translator\n---\n# Translator",
        manifest={
            "name": "translator",
            "slug": "translator",
            "version": "0.1.0",
            "description": "中英翻译",
            "runtime_type": "native",
            "tools": [
                {
                    "name": "translate",
                    "description": "翻译",
                    "parameters": {},
                    "required": [],
                    "entry": {"command": "python bundle/translate.py"},
                }
            ],
        },
        scripts=[ScriptFile(path="bundle/translate.py", content="print('ok')")],
        dependencies=["googletrans"],
    )

    mock_llm, _ = _make_mock_llm(
        structured_returns=[analyze_blueprint, generated],
    )
    service = SkillCreatorService(
        llm=mock_llm,
        github_client=mock_github,
        skill_service=mock_skill_service,
    )

    blueprint = await service.analyze("创建翻译工具")
    assert blueprint.skill_name == "translator"

    # Step 2: generate (with blueprint, skip analyze)
    generated_files = None
    progress_steps = []
    async for event in service.generate(
        description="创建翻译工具",
        sandbox=mock_sandbox,
        blueprint=blueprint,
    ):
        if isinstance(event, SkillGeneratedFiles):
            generated_files = event
        elif isinstance(event, SkillCreationProgress):
            progress_steps.append(event.step)

    assert generated_files is not None
    assert "analyzing" not in progress_steps
    assert "researching" in progress_steps
    assert "installing" not in progress_steps

    # Step 3: install
    result = await service.install(generated_files, installed_by="user-1")
    assert result.skill_name == "translator"
    assert "创建成功" in result.summary
    mock_skill_service.install_skill.assert_called_once()
