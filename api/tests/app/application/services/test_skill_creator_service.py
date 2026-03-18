from __future__ import annotations

import ast
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.skill_creator_service import (
    ManifestOutput,
    SkillCreatorService,
    _normalize_generated_dependencies,
    _normalize_generated_manifest,
)
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
    async def test_phased_generation_produces_valid_files(
        self,
        mock_github_client: MagicMock,
        mock_skill_service: AsyncMock,
    ) -> None:
        """Phased pipeline: structured manifest → per-tool scripts → SKILL.md."""
        blueprint = SkillBlueprint(
            skill_name="test-skill",
            description="测试用",
            tools=[ToolDef(name="run", description="运行", parameters=[])],
            search_keywords=[],
            estimated_deps=["requests"],
        )

        # Phase 1: with_structured_output returns ManifestOutput
        manifest_output = ManifestOutput(
            manifest={
                "name": "test-skill",
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
            dependencies=["requests"],
        )
        # Phase 2 (1 tool): ainvoke returns Python code
        script_code = "import sys\nimport json\nprint(json.dumps({'ok': True}))"
        # Phase 3: ainvoke returns SKILL.md
        skill_md = '---\nname: test-skill\nversion: "0.1.0"\n---\n# Test'

        mock_llm, _ = _make_mock_llm(
            single_return=manifest_output,
            direct_returns=[script_code, skill_md],
        )
        service = SkillCreatorService(
            llm=mock_llm,
            github_client=mock_github_client,
            skill_service=mock_skill_service,
        )

        files = await service._generate_files(blueprint, "调研报告内容")

        assert files.manifest["name"] == "test-skill"
        assert files.manifest["runtime_type"] == "native"
        assert len(files.scripts) == 1
        assert files.scripts[0].path == "bundle/run.py"
        assert "import sys" in files.scripts[0].content
        assert "---" in files.skill_md
        assert files.dependencies == ["requests"]


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
        # _analyze_requirement → blueprint, _generate_manifest → manifest
        manifest_output = ManifestOutput(
            manifest={
                "name": "test-skill",
                "runtime_type": "native",
                "tools": [
                    {"name": "run", "description": "执行", "parameters": {},
                     "required": [], "entry": {"command": "python bundle/run.py"}}
                ],
            },
            dependencies=[],
        )
        mock_llm, _ = _make_mock_llm(
            structured_returns=[analyze_blueprint, manifest_output],
            direct_returns=[
                "import sys\nprint('ok')",          # Phase 2: script
                "---\nname: test-skill\n---\n# Test",  # Phase 3: SKILL.md
            ],
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
        manifest_output = ManifestOutput(
            manifest={
                "name": "test-skill",
                "runtime_type": "native",
                "tools": [
                    {"name": "run", "description": "执行", "parameters": {},
                     "required": [], "entry": {"command": "python bundle/run.py"}}
                ],
            },
            dependencies=[],
        )
        mock_llm, _ = _make_mock_llm(
            single_return=manifest_output,
            direct_returns=["print('ok')", "---\nname: test-skill\n---\n# Test"],
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
        manifest_output = ManifestOutput(
            manifest={
                "name": "test-skill",
                "runtime_type": "native",
                "tools": [
                    {"name": "run", "description": "执行", "parameters": {},
                     "required": [], "entry": {"command": "python bundle/run.py"}}
                ],
            },
            dependencies=[],
        )
        mock_llm, _ = _make_mock_llm(
            structured_returns=[analyze_blueprint, manifest_output],
            direct_returns=["print('ok')", "---\nname: test-skill\n---\n# Test"],
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
        """Phase 2 keeps producing bad syntax → generate() yields failure."""
        blueprint = SkillBlueprint(
            skill_name="bad-skill",
            description="坏的",
            tools=[ToolDef(name="run", description="运行", parameters=[])],
            search_keywords=[],
            estimated_deps=[],
        )
        # Phase 1 succeeds, Phase 2 always returns bad Python
        manifest_output = ManifestOutput(
            manifest={
                "name": "bad-skill",
                "runtime_type": "native",
                "tools": [
                    {"name": "run", "description": "运行", "parameters": {},
                     "required": [], "entry": {"command": "python bundle/run.py"}}
                ],
            },
            dependencies=[],
        )
        bad_code = "def f(\n  broken"
        mock_llm, _ = _make_mock_llm(
            single_return=manifest_output,
            # All 3 retries of Phase 2 return bad syntax
            direct_returns=[bad_code, bad_code, bad_code],
        )
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


class TestGenerateManifest:
    async def test_structured_output_path(
        self,
        mock_github_client: MagicMock,
        mock_skill_service: AsyncMock,
    ) -> None:
        manifest_output = ManifestOutput(
            manifest={
                "name": "test-skill",
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
            dependencies=["requests"],
        )
        mock_llm, _ = _make_mock_llm(single_return=manifest_output)
        service = SkillCreatorService(
            llm=mock_llm,
            github_client=mock_github_client,
            skill_service=mock_skill_service,
        )

        result = await service._generate_manifest(SAMPLE_BLUEPRINT, "调研报告")

        assert result.manifest["name"] == "test-skill"
        assert result.dependencies == ["requests"]
        mock_llm.with_structured_output.assert_called_once()

    async def test_fallback_to_ainvoke(
        self,
        mock_github_client: MagicMock,
        mock_skill_service: AsyncMock,
    ) -> None:
        mock_llm, structured = _make_mock_llm(
            direct_return={
                "manifest": {"name": "fallback", "runtime_type": "native", "tools": []},
                "dependencies": ["requests"],
            },
        )
        structured.ainvoke = AsyncMock(side_effect=Exception("tool calling not supported"))

        service = SkillCreatorService(
            llm=mock_llm,
            github_client=mock_github_client,
            skill_service=mock_skill_service,
        )

        result = await service._generate_manifest(SAMPLE_BLUEPRINT, "调研报告")

        assert result.manifest["name"] == "fallback"


class TestGenerateScript:
    async def test_generates_valid_python_script(
        self,
        mock_github_client: MagicMock,
        mock_skill_service: AsyncMock,
    ) -> None:
        code = 'import sys\nimport json\ndata = json.loads(sys.argv[1])\nprint(json.dumps({"ok": True}))'
        mock_llm, _ = _make_mock_llm(direct_return=f"```python\n{code}\n```")
        service = SkillCreatorService(
            llm=mock_llm,
            github_client=mock_github_client,
            skill_service=mock_skill_service,
        )

        tool_def = {
            "name": "parse_url",
            "description": "解析URL",
            "parameters": {"url": {"type": "string", "description": "视频URL"}},
            "required": ["url"],
            "entry": {"command": "python bundle/parse_url.py"},
        }
        result = await service._generate_script(tool_def, SAMPLE_BLUEPRINT, "调研报告")

        assert result.path == "bundle/parse_url.py"
        assert "import sys" in result.content
        ast.parse(result.content)

    async def test_fallback_path_from_tool_name(
        self,
        mock_github_client: MagicMock,
        mock_skill_service: AsyncMock,
    ) -> None:
        mock_llm, _ = _make_mock_llm(direct_return="print('ok')")
        service = SkillCreatorService(
            llm=mock_llm,
            github_client=mock_github_client,
            skill_service=mock_skill_service,
        )

        tool_def = {"name": "my_tool", "description": "d"}
        result = await service._generate_script(tool_def, SAMPLE_BLUEPRINT, "")

        assert result.path == "bundle/my_tool.py"


class TestGenerateSkillMd:
    async def test_generates_valid_skill_md(
        self,
        mock_github_client: MagicMock,
        mock_skill_service: AsyncMock,
    ) -> None:
        skill_md_content = '---\nname: test-skill\nversion: "0.1.0"\n---\n\n# Test Skill\n\nUsage docs.'
        mock_llm, _ = _make_mock_llm(direct_return=skill_md_content)
        service = SkillCreatorService(
            llm=mock_llm,
            github_client=mock_github_client,
            skill_service=mock_skill_service,
        )

        manifest = {"name": "test-skill", "tools": [{"name": "run"}]}
        scripts = [ScriptFile(path="bundle/run.py", content="print('ok')")]
        result = await service._generate_skill_md(manifest, scripts, SAMPLE_BLUEPRINT)

        assert "---" in result
        assert "test-skill" in result

    async def test_strips_wrapping_code_fences(
        self,
        mock_github_client: MagicMock,
        mock_skill_service: AsyncMock,
    ) -> None:
        wrapped = "```yaml\n---\nname: t\n---\n# T\n```"
        mock_llm, _ = _make_mock_llm(direct_return=wrapped)
        service = SkillCreatorService(
            llm=mock_llm,
            github_client=mock_github_client,
            skill_service=mock_skill_service,
        )

        result = await service._generate_skill_md({}, [], SAMPLE_BLUEPRINT)
        assert result.startswith("---")
        assert "```" not in result


class TestFixFiles:
    async def test_script_error_only_regenerates_broken_script(
        self,
        mock_github_client: MagicMock,
        mock_skill_service: AsyncMock,
    ) -> None:
        existing = SkillGeneratedFiles(
            skill_md="---\nname: t\n---\n# T",
            manifest={
                "name": "t",
                "runtime_type": "native",
                "tools": [
                    {"name": "good", "description": "ok", "entry": {"command": "python bundle/good.py"}},
                    {"name": "bad", "description": "broken", "entry": {"command": "python bundle/bad.py"}},
                ],
            },
            scripts=[
                ScriptFile(path="bundle/good.py", content="print('ok')"),
                ScriptFile(path="bundle/bad.py", content="broken code"),
            ],
            dependencies=[],
        )
        errors = ["脚本 --help 失败 (bundle/bad.py): SyntaxError"]

        fixed_code = "import sys\nprint('fixed')"
        mock_llm, _ = _make_mock_llm(direct_return=fixed_code)
        service = SkillCreatorService(
            llm=mock_llm,
            github_client=mock_github_client,
            skill_service=mock_skill_service,
        )

        result = await service._fix_files(existing, errors, SAMPLE_BLUEPRINT, "报告")

        good_script = next(s for s in result.scripts if s.path == "bundle/good.py")
        bad_script = next(s for s in result.scripts if s.path == "bundle/bad.py")
        assert good_script.content == "print('ok')"
        assert "fixed" in bad_script.content

    async def test_manifest_error_triggers_full_regeneration(
        self,
        mock_github_client: MagicMock,
        mock_skill_service: AsyncMock,
    ) -> None:
        existing = SkillGeneratedFiles(
            skill_md="---\nname: t\n---\n# T",
            manifest={"name": "t", "runtime_type": "native", "tools": []},
            scripts=[ScriptFile(path="bundle/run.py", content="print('ok')")],
            dependencies=[],
        )
        errors = ["manifest 缺少必填字段: tools"]

        manifest_output = ManifestOutput(
            manifest={"name": "t", "runtime_type": "native", "tools": [
                {"name": "run", "description": "d", "entry": {"command": "python bundle/run.py"}}
            ]},
            dependencies=[],
        )
        mock_llm, _ = _make_mock_llm(
            single_return=manifest_output,
            direct_returns=["print('regen')", "---\nname: t\n---\n# T"],
        )
        service = SkillCreatorService(
            llm=mock_llm,
            github_client=mock_github_client,
            skill_service=mock_skill_service,
        )

        result = await service._fix_files(existing, errors, SAMPLE_BLUEPRINT, "报告")

        mock_llm.with_structured_output.assert_called()


class TestManifestOutput:
    def test_basic_creation(self) -> None:
        mo = ManifestOutput(
            manifest={"name": "test", "runtime_type": "native", "tools": []},
            dependencies=["requests"],
        )
        assert mo.manifest["name"] == "test"
        assert mo.dependencies == ["requests"]

    def test_extra_fields_ignored(self) -> None:
        mo = ManifestOutput.model_validate({
            "manifest": {"name": "t", "tools": []},
            "dependencies": [],
            "extra_field": "ignored",
        })
        assert mo.manifest["name"] == "t"


class TestNormalizeGeneratedManifest:
    def test_auto_fills_runtime_type(self) -> None:
        manifest = {"name": "test", "tools": []}
        result = _normalize_generated_manifest(manifest)
        assert result["runtime_type"] == "native"

    def test_auto_fills_entry_command(self) -> None:
        manifest = {
            "name": "test",
            "runtime_type": "native",
            "tools": [{"name": "run", "description": "runs"}],
        }
        result = _normalize_generated_manifest(manifest)
        assert result["tools"][0]["entry"]["command"] == "python bundle/run.py"

    def test_preserves_existing_entry(self) -> None:
        manifest = {
            "name": "test",
            "runtime_type": "native",
            "tools": [
                {"name": "run", "description": "runs", "entry": {"command": "python bundle/custom.py"}}
            ],
        }
        result = _normalize_generated_manifest(manifest)
        assert result["tools"][0]["entry"]["command"] == "python bundle/custom.py"


class TestNormalizeGeneratedDependencies:
    def test_dict_to_list(self) -> None:
        raw = {"pip_packages": ["requests", "yt-dlp"], "capabilities": ["中文处理"]}
        result = _normalize_generated_dependencies(raw)
        assert result == ["requests", "yt-dlp"]

    def test_list_of_dicts(self) -> None:
        raw = [{"name": "requests"}, {"name": "中文包"}]
        result = _normalize_generated_dependencies(raw)
        assert result == ["requests"]

    def test_normal_list_unchanged(self) -> None:
        raw = ["requests", "beautifulsoup4"]
        result = _normalize_generated_dependencies(raw)
        assert result == ["requests", "beautifulsoup4"]
