from __future__ import annotations

import ast
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.skill_creator_service import SkillCreatorService
from app.domain.models.skill_creator import (
    SkillBlueprint,
    SkillCreationProgress,
    SkillCreationResult,
    SkillGeneratedFiles,
    ScriptFile,
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
    """Build a mock BaseChatModel that supports both call patterns:

    1. llm.with_structured_output(schema).ainvoke(messages) — for _analyze_requirement
    2. llm.ainvoke(messages) — for _generate_files (returns AIMessage-like with .content)

    *structured_returns* / *single_return* control with_structured_output path.
    *direct_returns* / *direct_return* control direct ainvoke path. If a direct
    value is a Pydantic BaseModel or dict, it is auto-serialized to JSON .content.
    """
    llm = MagicMock()
    structured_runnable = AsyncMock()

    if structured_returns is not None:
        structured_runnable.ainvoke = AsyncMock(side_effect=structured_returns)
    elif single_return is not None:
        structured_runnable.ainvoke = AsyncMock(return_value=single_return)
    else:
        structured_runnable.ainvoke = AsyncMock()

    llm.with_structured_output = MagicMock(return_value=structured_runnable)

    # Direct ainvoke — wrap values as SimpleNamespace(content=json_str)
    def _to_ai_message(val):
        if hasattr(val, "model_dump_json"):
            return SimpleNamespace(content=val.model_dump_json())
        if isinstance(val, dict):
            import json as _json
            return SimpleNamespace(content=_json.dumps(val, ensure_ascii=False))
        if isinstance(val, str):
            return SimpleNamespace(content=val)
        return val

    if direct_returns is not None:
        llm.ainvoke = AsyncMock(side_effect=[_to_ai_message(v) for v in direct_returns])
    elif direct_return is not None:
        llm.ainvoke = AsyncMock(return_value=_to_ai_message(direct_return))
    else:
        # Auto: if structured_returns has >1 item, use the 2nd+ as direct returns
        # (1st is used by with_structured_output for _analyze_requirement)
        llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content="{}"))

    return llm, structured_runnable


@pytest.fixture
def mock_github_client() -> MagicMock:
    client = MagicMock()
    client.research_keywords = AsyncMock(return_value=[])
    client.format_research_report = MagicMock(return_value="未找到相关参考。")
    return client


@pytest.fixture
def mock_skill_service() -> AsyncMock:
    service = AsyncMock()
    service.install_skill = AsyncMock(
        return_value=SimpleNamespace(id="test-skill--abc12345", name="test-skill")
    )
    return service


@pytest.fixture
def mock_sandbox() -> AsyncMock:
    sandbox = AsyncMock()
    sandbox.id = "sandbox-123"
    sandbox.exec_command = AsyncMock(
        return_value=ToolResult(success=True, message='{"result": "ok"}'),
    )
    sandbox.write_file = AsyncMock(return_value=ToolResult(success=True, message="OK"))
    return sandbox


# --------------- Prebuilt model objects for tests --------------- #

SAMPLE_BLUEPRINT = SkillBlueprint(
    skill_name="bilibili-whisper-summary",
    description="下载B站视频并用whisper转录后总结",
    tools=[
        ToolDef(
            name="download_and_summarize",
            description="下载视频并总结",
            parameters=[],
        )
    ],
    search_keywords=["bilibili download python", "whisper transcribe"],
    estimated_deps=["yt-dlp", "openai-whisper"],
)

SAMPLE_GENERATED_FILES = SkillGeneratedFiles(
    skill_md="---\nname: test-skill\n---\n# Test Skill",
    manifest={
        "name": "test-skill",
        "slug": "test-skill",
        "version": "0.1.0",
        "description": "测试用",
        "runtime_type": "native",
        "tools": [
            {
                "name": "run",
                "description": "运行",
                "parameters": {},
                "required": [],
                "entry": {"command": "python bundle/run.py"},
            }
        ],
    },
    scripts=[ScriptFile(path="bundle/run.py", content="import sys\nprint('hello')")],
    dependencies=["requests"],
)


class TestAnalyzeRequirement:
    async def test_analyze_extracts_blueprint(
        self,
        mock_github_client: MagicMock,
        mock_skill_service: AsyncMock,
    ) -> None:
        blueprint = SkillBlueprint(
            skill_name="bilibili-whisper-summary",
            description="下载B站视频并用whisper转录后总结",
            tools=[
                ToolDef(
                    name="download_and_summarize",
                    description="下载视频并总结",
                    parameters=[],
                )
            ],
            search_keywords=["bilibili download python", "whisper transcribe"],
            estimated_deps=["yt-dlp", "openai-whisper"],
        )
        mock_llm, _ = _make_mock_llm(single_return=blueprint)
        service = SkillCreatorService(
            llm=mock_llm,
            github_client=mock_github_client,
            skill_service=mock_skill_service,
        )

        result = await service._analyze_requirement("帮我创建一个下载B站视频的 skill")

        assert result.skill_name == "bilibili-whisper-summary"
        assert len(result.tools) == 1
        assert "yt-dlp" in result.estimated_deps

    async def test_analyze_public_method(
        self,
        mock_github_client: MagicMock,
        mock_skill_service: AsyncMock,
    ) -> None:
        blueprint = SkillBlueprint(
            skill_name="translator",
            description="翻译工具",
            tools=[ToolDef(name="translate", description="翻译", parameters=[])],
            search_keywords=["translate python"],
            estimated_deps=["googletrans"],
        )
        mock_llm, _ = _make_mock_llm(single_return=blueprint)
        service = SkillCreatorService(
            llm=mock_llm,
            github_client=mock_github_client,
            skill_service=mock_skill_service,
        )

        result = await service.analyze("创建一个翻译工具")
        assert result.skill_name == "translator"
        assert len(result.tools) == 1


class TestGenerateFiles:
    async def test_generate_produces_valid_files(
        self,
        mock_github_client: MagicMock,
        mock_skill_service: AsyncMock,
    ) -> None:
        blueprint = SkillBlueprint(
            skill_name="test-skill",
            description="测试用",
            tools=[ToolDef(name="run", description="运行", parameters=[])],
            search_keywords=[],
            estimated_deps=["requests"],
        )

        generated = SkillGeneratedFiles(
            skill_md="---\nname: test-skill\n---\n# Test Skill",
            manifest={
                "name": "test-skill",
                "slug": "test-skill",
                "version": "0.1.0",
                "description": "测试用",
                "runtime_type": "native",
                "tools": [
                    {
                        "name": "run",
                        "description": "运行",
                        "parameters": {},
                        "required": [],
                        "entry": {"command": "python bundle/run.py"},
                    }
                ],
            },
            scripts=[ScriptFile(path="bundle/run.py", content="import sys\nprint('hello')")],
            dependencies=["requests"],
        )
        mock_llm, _ = _make_mock_llm(single_return=generated)
        service = SkillCreatorService(
            llm=mock_llm,
            github_client=mock_github_client,
            skill_service=mock_skill_service,
        )

        files = await service._generate_files(blueprint, "调研报告内容")

        assert "name: test-skill" in files.skill_md
        assert files.manifest["runtime_type"] == "native"
        assert len(files.scripts) == 1
        ast.parse(files.scripts[0].content)


class TestValidateInSandbox:
    async def test_validate_passes_on_valid_script(
        self,
        mock_github_client: MagicMock,
        mock_skill_service: AsyncMock,
        mock_sandbox: AsyncMock,
    ) -> None:
        mock_llm, _ = _make_mock_llm()
        service = SkillCreatorService(
            llm=mock_llm,
            github_client=mock_github_client,
            skill_service=mock_skill_service,
        )
        files = SkillGeneratedFiles(
            skill_md="---\nname: t\n---\n# T",
            manifest={"name": "t", "tools": []},
            scripts=[ScriptFile(path="bundle/run.py", content="print('ok')")],
            dependencies=["requests"],
        )

        errors = await service._validate_in_sandbox(files, mock_sandbox)

        assert errors == []

    async def test_validate_catches_syntax_error(
        self,
        mock_github_client: MagicMock,
        mock_skill_service: AsyncMock,
        mock_sandbox: AsyncMock,
    ) -> None:
        mock_llm, _ = _make_mock_llm()
        service = SkillCreatorService(
            llm=mock_llm,
            github_client=mock_github_client,
            skill_service=mock_skill_service,
        )
        files = SkillGeneratedFiles(
            skill_md="---\nname: t\n---\n# T",
            manifest={"name": "t", "tools": []},
            scripts=[ScriptFile(path="bundle/run.py", content="def f(\n  broken")],
            dependencies=[],
        )

        errors = await service._validate_in_sandbox(files, mock_sandbox)

        assert errors
        assert "语法错误" in errors[0] or "SyntaxError" in errors[0]


class TestFullPipeline:
    async def test_create_yields_progress_events(
        self,
        mock_github_client: MagicMock,
        mock_skill_service: AsyncMock,
        mock_sandbox: AsyncMock,
    ) -> None:
        analyze_blueprint = SkillBlueprint(
            skill_name="test-skill",
            description="测试",
            tools=[ToolDef(name="run", description="执行", parameters=[])],
            search_keywords=["test python"],
            estimated_deps=[],
        )
        generate_files = SkillGeneratedFiles(
            skill_md="---\nname: test-skill\n---\n# Test",
            manifest={
                "name": "test-skill",
                "slug": "test-skill",
                "version": "0.1.0",
                "description": "测试",
                "runtime_type": "native",
                "tools": [
                    {
                        "name": "run",
                        "description": "执行",
                        "parameters": {},
                        "required": [],
                        "entry": {"command": "python bundle/run.py"},
                    }
                ],
            },
            scripts=[ScriptFile(path="bundle/run.py", content="print('ok')")],
            dependencies=[],
        )
        mock_llm, _ = _make_mock_llm(
            structured_returns=[analyze_blueprint, generate_files],
        )
        service = SkillCreatorService(
            llm=mock_llm,
            github_client=mock_github_client,
            skill_service=mock_skill_service,
        )

        events = []
        async for event in service.create(
            description="创建一个测试 skill",
            sandbox=mock_sandbox,
            installed_by="user-1",
        ):
            events.append(event)

        progress_steps = [
            event.step
            for event in events
            if isinstance(event, SkillCreationProgress)
        ]
        results = [
            event
            for event in events
            if isinstance(event, SkillCreationResult)
        ]

        assert "analyzing" in progress_steps
        assert "researching" in progress_steps
        assert "generating" in progress_steps
        assert "validating" in progress_steps
        assert "installing" in progress_steps
        assert len(results) == 1
        assert results[0].skill_name == "test-skill"


class TestGenerate:
    """Tests for the public generate() method."""

    async def test_generate_returns_files_on_success(
        self,
        mock_github_client: MagicMock,
        mock_skill_service: AsyncMock,
        mock_sandbox: AsyncMock,
    ) -> None:
        """With blueprint provided, returns SkillGeneratedFiles, no 'installing' step."""
        blueprint = SkillBlueprint(
            skill_name="test-skill",
            description="测试",
            tools=[ToolDef(name="run", description="执行", parameters=[])],
            search_keywords=["test python"],
            estimated_deps=[],
        )
        generated = SkillGeneratedFiles(
            skill_md="---\nname: test-skill\n---\n# Test",
            manifest={
                "name": "test-skill",
                "slug": "test-skill",
                "version": "0.1.0",
                "description": "测试",
                "runtime_type": "native",
                "tools": [
                    {
                        "name": "run",
                        "description": "执行",
                        "parameters": {},
                        "required": [],
                        "entry": {"command": "python bundle/run.py"},
                    }
                ],
            },
            scripts=[ScriptFile(path="bundle/run.py", content="print('ok')")],
            dependencies=[],
        )
        mock_llm, _ = _make_mock_llm(single_return=generated)
        service = SkillCreatorService(
            llm=mock_llm,
            github_client=mock_github_client,
            skill_service=mock_skill_service,
        )

        events: list = []
        async for event in service.generate(
            description="创建一个测试 skill",
            sandbox=mock_sandbox,
            blueprint=blueprint,
        ):
            events.append(event)

        progress_steps = [
            e.step for e in events if isinstance(e, SkillCreationProgress)
        ]
        file_results = [e for e in events if isinstance(e, SkillGeneratedFiles)]

        # Should NOT have analyzing or installing steps
        assert "analyzing" not in progress_steps
        assert "installing" not in progress_steps
        # Should have researching, generating, validating
        assert "researching" in progress_steps
        assert "generating" in progress_steps
        assert "validating" in progress_steps
        # Should yield SkillGeneratedFiles at the end
        assert len(file_results) == 1
        assert file_results[0].manifest["name"] == "test-skill"

    async def test_generate_without_blueprint_runs_analyze(
        self,
        mock_github_client: MagicMock,
        mock_skill_service: AsyncMock,
        mock_sandbox: AsyncMock,
    ) -> None:
        """Without blueprint, includes 'analyzing' step."""
        analyze_blueprint = SkillBlueprint(
            skill_name="test-skill",
            description="测试",
            tools=[ToolDef(name="run", description="执行", parameters=[])],
            search_keywords=["test python"],
            estimated_deps=[],
        )
        generated = SkillGeneratedFiles(
            skill_md="---\nname: test-skill\n---\n# Test",
            manifest={
                "name": "test-skill",
                "slug": "test-skill",
                "version": "0.1.0",
                "description": "测试",
                "runtime_type": "native",
                "tools": [
                    {
                        "name": "run",
                        "description": "执行",
                        "parameters": {},
                        "required": [],
                        "entry": {"command": "python bundle/run.py"},
                    }
                ],
            },
            scripts=[ScriptFile(path="bundle/run.py", content="print('ok')")],
            dependencies=[],
        )
        mock_llm, _ = _make_mock_llm(
            structured_returns=[analyze_blueprint, generated],
        )
        service = SkillCreatorService(
            llm=mock_llm,
            github_client=mock_github_client,
            skill_service=mock_skill_service,
        )

        events: list = []
        async for event in service.generate(
            description="创建一个测试 skill",
            sandbox=mock_sandbox,
            blueprint=None,
        ):
            events.append(event)

        progress_steps = [
            e.step for e in events if isinstance(e, SkillCreationProgress)
        ]
        assert "analyzing" in progress_steps
        assert "installing" not in progress_steps

    async def test_generate_returns_validation_errors(
        self,
        mock_github_client: MagicMock,
        mock_skill_service: AsyncMock,
        mock_sandbox: AsyncMock,
    ) -> None:
        """Syntax error in script yields validation failure progress."""
        blueprint = SkillBlueprint(
            skill_name="bad-skill",
            description="坏的",
            tools=[ToolDef(name="run", description="运行", parameters=[])],
            search_keywords=[],
            estimated_deps=[],
        )
        bad_files = SkillGeneratedFiles(
            skill_md="---\nname: bad-skill\n---\n# Bad",
            manifest={"name": "bad-skill", "tools": []},
            scripts=[
                ScriptFile(path="bundle/run.py", content="def f(\n  broken")
            ],
            dependencies=[],
        )
        # generate_files called once, then _fix_files calls generate_files again (MAX_FIX_RETRIES=2)
        mock_llm, _ = _make_mock_llm(single_return=bad_files)
        service = SkillCreatorService(
            llm=mock_llm,
            github_client=mock_github_client,
            skill_service=mock_skill_service,
        )

        events: list = []
        async for event in service.generate(
            description="创建坏的 skill",
            sandbox=mock_sandbox,
            blueprint=blueprint,
        ):
            events.append(event)

        progress_events = [
            e for e in events if isinstance(e, SkillCreationProgress)
        ]
        file_results = [e for e in events if isinstance(e, SkillGeneratedFiles)]

        # Should NOT yield files on failure
        assert len(file_results) == 0
        # Last progress should indicate failure
        last_progress = progress_events[-1]
        assert "失败" in last_progress.message


class TestInstallPublic:
    """Tests for the public install() method."""

    async def test_install_returns_creation_result(
        self,
        mock_github_client: MagicMock,
        mock_skill_service: AsyncMock,
    ) -> None:
        mock_llm, _ = _make_mock_llm()
        service = SkillCreatorService(
            llm=mock_llm,
            github_client=mock_github_client,
            skill_service=mock_skill_service,
        )
        files = SkillGeneratedFiles(
            skill_md="---\nname: test-skill\n---\n# Test",
            manifest={
                "name": "test-skill",
                "tools": [
                    {"name": "run", "description": "执行"},
                    {"name": "check", "description": "检查"},
                ],
            },
            scripts=[
                ScriptFile(path="bundle/run.py", content="print('ok')"),
                ScriptFile(path="bundle/check.py", content="print('check')"),
            ],
            dependencies=[],
        )

        result = await service.install(files, installed_by="user-1")

        assert isinstance(result, SkillCreationResult)
        assert result.skill_id == "test-skill--abc12345"
        assert result.skill_name == "test-skill"
        assert result.tools == ["run", "check"]
        # 2 scripts + 2 (SKILL.md + manifest.json)
        assert result.files_count == 4
        assert "创建成功" in result.summary
        mock_skill_service.install_skill.assert_awaited_once()
