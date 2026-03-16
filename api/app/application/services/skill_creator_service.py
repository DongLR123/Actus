"""Skill Creator Service — AI 驱动 Skill 创建流水线。"""

from __future__ import annotations

import ast
import json
import logging
import re
import shlex
import tempfile
from pathlib import Path
from typing import AsyncGenerator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.domain.external.sandbox import Sandbox
from app.domain.models.skill import SkillSourceType
from app.domain.models.skill_creator import (
    GitHubRepoInfo,
    ScriptFile,
    SkillBlueprint,
    SkillCreationProgress,
    SkillCreationResult,
    SkillGeneratedFiles,
)
from app.infrastructure.external.github_search_client import GitHubSearchClient

logger = logging.getLogger(__name__)

MAX_FIX_RETRIES = 2

ANALYZE_SYSTEM_PROMPT = """\
你是 Skill 架构师。你的任务是：将用户的"创建 Skill"需求解析为结构化 JSON 蓝图。

⚠️ 重要区分：你在**设计 Skill 的架构蓝图**，而不是执行 Skill 描述的功能。
例如用户说"创建一个翻译工具 Skill"，你应该输出翻译工具的架构蓝图（包含工具名、参数定义等），
而不是执行翻译操作。用户消息中的示例数据（如 URL、文本）仅用于理解需求，不要把它们当作实际输入。

必须严格遵循以下 JSON 结构（注意字段名必须完全一致）：
{
  "skill_name": "english-kebab-case-name",
  "description": "该 Skill 的功能描述",
  "tools": [
    {
      "name": "tool_function_name",
      "description": "工具功能描述",
      "parameters": [
        {"name": "param1", "type": "string", "description": "参数说明", "required": true}
      ]
    }
  ],
  "search_keywords": ["keyword1", "keyword2"],
  "estimated_deps": ["requests", "beautifulsoup4"]
}

关键约束：
- 顶层字段必须是 skill_name（不是 name/skill/title），使用英文 kebab-case
- 直接返回 JSON 对象，不要嵌套在其他键下
- 仅返回上述结构的 JSON，不要返回其他文本或执行实际操作"""

GENERATE_SYSTEM_PROMPT = """\
你是 Skill 代码生成器。请根据需求蓝图和 GitHub 调研结果，生成可安装的 Native Skill。

必须返回严格 JSON，包含以下四个字段：

### 1. skill_md (string)
完整的 SKILL.md 内容，格式必须严格如下（注意 --- 分隔符）：
```
---
name: skill-name
version: "0.1.0"
description: 功能描述
runtime_type: native
tools:
  - name: tool_name
    description: 工具描述
    parameters:
      param1:
        type: string
        description: 参数描述
    required:
      - param1
    entry:
      command: python bundle/tool_name.py
activation: {}
policy:
  risk_level: low
---

# Skill 名称

使用说明文档...
```

### 2. manifest (object)
必须符合以下结构：
{
  "name": "skill-name",
  "slug": "skill-name",
  "version": "0.1.0",
  "description": "功能描述",
  "runtime_type": "native",
  "tools": [
    {
      "name": "tool_name",
      "description": "工具描述",
      "parameters": {
        "param1": {"type": "string", "description": "参数描述"}
      },
      "required": ["param1"],
      "entry": {
        "command": "python bundle/tool_name.py"
      }
    }
  ],
  "activation": {},
  "policy": {"risk_level": "low"},
  "security": {}
}
注意：
- parameters 必须是 object 格式（键为参数名，值为 {type, description}），不是数组
- entry.command 指向 bundle/ 下的脚本，格式为 "python bundle/<script>.py"
- runtime_type 必须为 "native"

### 3. scripts (array)
每个元素包含 path 和 content：
[{"path": "bundle/tool_name.py", "content": "...python code..."}]

脚本要求：
- 接受一个 JSON 字符串作为命令行参数: sys.argv[1]
- 输出 JSON 格式结果到 stdout
- 支持 --help 参数（用 argparse 或简单判断）
- 优先使用调研到的成熟开源库

### 4. dependencies (array)
需要 pip install 的包名列表。

只返回 JSON，不要返回其他文字。"""


def _parse_llm_json(text: str) -> dict:
    """从 LLM 文本响应中提取 JSON 对象。

    处理常见格式：
    - 纯 JSON
    - ```json ... ``` markdown 代码块
    - 混合文本中的 JSON 对象
    """
    text = text.strip()

    # 1. 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. 提取 markdown 代码块中的 JSON
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if md_match:
        try:
            return json.loads(md_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. 找到第一个 { 到最后一个 } 之间的内容
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace : last_brace + 1])
        except json.JSONDecodeError:
            pass

    # 4. 使用 json_repair 库兜底
    try:
        from json_repair import repair_json
        repaired = repair_json(text, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
    except Exception:
        pass

    raise ValueError(f"无法从 LLM 响应中提取 JSON（前 200 字符）: {text[:200]}")


def _validate_generated_files(files: SkillGeneratedFiles) -> None:
    """校验 LLM 生成结果的结构完整性。

    检查 LLM 是否真正生成了可执行的 Skill 文件，而不是仅返回文档或 JSON 描述符。
    """
    errors: list[str] = []

    # 1. manifest 不能是空壳（LLM 未生成 manifest 时会 fallback 为 {}）
    if not files.manifest or not any(
        k in files.manifest for k in ("name", "tools", "runtime_type")
    ):
        errors.append("manifest 为空或缺少关键字段（name/tools/runtime_type）")

    # 2. 必须有 bundle/ 下的 Python 脚本
    py_scripts = [s for s in files.scripts if s.path.endswith(".py")]
    bundle_scripts = [s for s in py_scripts if s.path.startswith("bundle/")]
    if not bundle_scripts:
        errors.append(
            f"缺少 bundle/*.py 可执行脚本（当前 scripts: "
            f"{[s.path for s in files.scripts] or '无'}）"
        )

    # 3. dependencies 应为 pip 包名，不应为中文描述
    non_pip_deps = [
        d for d in files.dependencies
        if any('\u4e00' <= c <= '\u9fff' for c in d)
    ]
    if non_pip_deps:
        errors.append(
            f"dependencies 应为 pip 包名，不应为中文描述: {non_pip_deps[:3]}"
        )

    # 4. skill_md 不应为空
    if not files.skill_md.strip():
        errors.append("skill_md 为空")

    if errors:
        raise ValueError(
            f"LLM 生成结果不符合 Skill 结构要求（{len(errors)} 个问题）：\n"
            + "\n".join(f"- {e}" for e in errors)
        )


_SAMPLE_VALUES = {
    "string": "test",
    "str": "test",
    "number": 1,
    "integer": 1,
    "int": 1,
    "float": 1.0,
    "boolean": True,
    "bool": True,
    "array": [],
    "list": [],
    "object": {},
    "dict": {},
}


def _build_sample_input(tool_def: dict) -> dict:
    """根据 manifest tool 定义构造最小测试 JSON 输入。"""
    params_raw = tool_def.get("parameters", {})
    required_raw = tool_def.get("required", [])
    required_names = set(required_raw) if isinstance(required_raw, list) else set()

    sample: dict = {}

    # parameters 可能是 {"param": {type, desc}} 或 [{name, type, desc, required}]
    if isinstance(params_raw, dict):
        for name, spec in params_raw.items():
            if not isinstance(spec, dict):
                continue
            param_type = str(spec.get("type", "string")).lower()
            sample[name] = _SAMPLE_VALUES.get(param_type, "test")
    elif isinstance(params_raw, list):
        for spec in params_raw:
            if not isinstance(spec, dict):
                continue
            name = spec.get("name", "")
            if not name:
                continue
            param_type = str(spec.get("type", "string")).lower()
            sample[name] = _SAMPLE_VALUES.get(param_type, "test")

    return sample


class SkillCreatorService:
    """五步流水线：分析、调研、生成、验证、安装。"""

    def __init__(
        self,
        llm: BaseChatModel,
        github_client: GitHubSearchClient,
        skill_service,
    ) -> None:
        self._llm = llm
        self._github = github_client
        self._skill_service = skill_service

    async def create(
        self,
        description: str,
        sandbox: Sandbox | None = None,
        installed_by: str = "",
    ) -> AsyncGenerator[SkillCreationProgress | SkillCreationResult, None]:
        from app.infrastructure.external.sandbox.docker_sandbox import DockerSandbox

        temp_sandbox: Sandbox | None = None

        try:
            yield SkillCreationProgress(step="analyzing", message="正在分析需求...")
            blueprint = await self._analyze_requirement(description)

            yield SkillCreationProgress(
                step="researching",
                message="正在调研 GitHub 方案...",
            )
            repos = await self._research(blueprint)
            report = self._github.format_research_report(repos)
            yield SkillCreationProgress(
                step="researching",
                message=f"调研完成，找到 {len(repos)} 个参考仓库",
                references=repos or None,
            )

            yield SkillCreationProgress(step="generating", message="正在生成 Skill 文件...")
            files = await self._generate_files(blueprint, report)
            yield SkillCreationProgress(
                step="generating",
                message=f"生成完成，共 {len(files.scripts)} 个脚本",
            )

            yield SkillCreationProgress(step="validating", message="正在执行沙箱验证...")
            if sandbox is None:
                temp_sandbox = await DockerSandbox.create()
                sandbox = temp_sandbox

            errors = await self._validate_in_sandbox(files, sandbox)
            fix_round = 0
            while errors and fix_round < MAX_FIX_RETRIES:
                yield SkillCreationProgress(
                    step="validating",
                    message=f"检测到问题，正在自动修复（第 {fix_round + 1} 次）...",
                    detail=errors[0],
                )
                files = await self._fix_files(files, errors, blueprint, report)
                errors = await self._validate_in_sandbox(files, sandbox)
                fix_round += 1

            if errors:
                yield SkillCreationProgress(
                    step="validating",
                    message="沙箱验证失败",
                    detail=errors[0],
                )
                return

            yield SkillCreationProgress(step="validating", message="沙箱验证通过")

            yield SkillCreationProgress(step="installing", message="正在安装 Skill...")
            installed_skill = await self._install(files, installed_by)
            tool_names = [
                str(tool.get("name") or "")
                for tool in files.manifest.get("tools", [])
                if isinstance(tool, dict)
            ]
            yield SkillCreationProgress(step="installing", message="安装完成")
            yield SkillCreationResult(
                skill_id=installed_skill.id,
                skill_name=installed_skill.name,
                tools=tool_names,
                files_count=len(files.scripts) + 2,
                summary=f"Skill '{installed_skill.name}' 创建成功",
            )
        finally:
            if temp_sandbox:
                try:
                    await temp_sandbox.destroy()
                except Exception as exc:
                    logger.warning("销毁临时沙箱失败: %s", exc)

    async def analyze(self, description: str) -> SkillBlueprint:
        """公共分析接口，供 brainstorm_skill 工具调用。"""
        return await self._analyze_requirement(description)

    async def generate(
        self,
        description: str,
        sandbox: Sandbox,
        blueprint: SkillBlueprint | None = None,
    ) -> AsyncGenerator[SkillCreationProgress | SkillGeneratedFiles, None]:
        """执行 RESEARCHING → GENERATING → VALIDATING，返回生成文件，不安装。"""
        if blueprint is None:
            yield SkillCreationProgress(step="analyzing", message="正在分析需求...")
            blueprint = await self._analyze_requirement(description)

        yield SkillCreationProgress(step="researching", message="正在调研 GitHub 方案...")
        repos = await self._research(blueprint)
        report = self._github.format_research_report(repos)
        yield SkillCreationProgress(
            step="researching",
            message=f"调研完成，找到 {len(repos)} 个参考仓库",
            references=repos or None,
        )

        yield SkillCreationProgress(step="generating", message="正在生成 Skill 文件...")
        files = await self._generate_files(blueprint, report)
        yield SkillCreationProgress(
            step="generating",
            message=f"生成完成，共 {len(files.scripts)} 个脚本",
        )

        yield SkillCreationProgress(step="validating", message="正在执行沙箱验证...")
        errors = await self._validate_in_sandbox(files, sandbox)
        fix_round = 0
        while errors and fix_round < MAX_FIX_RETRIES:
            yield SkillCreationProgress(
                step="validating",
                message=f"检测到问题，正在自动修复（第 {fix_round + 1} 次）...",
                detail=errors[0],
            )
            files = await self._fix_files(files, errors, blueprint, report)
            errors = await self._validate_in_sandbox(files, sandbox)
            fix_round += 1

        if errors:
            yield SkillCreationProgress(
                step="validating",
                message="沙箱验证失败",
                detail=errors[0],
            )
            return

        yield SkillCreationProgress(step="validating", message="沙箱验证通过")
        yield files

    async def install(self, files: SkillGeneratedFiles, installed_by: str) -> SkillCreationResult:
        """公共安装接口，供 install_skill 工具调用。"""
        installed_skill = await self._install(files, installed_by)
        tool_names = [
            str(tool.get("name") or "")
            for tool in files.manifest.get("tools", [])
            if isinstance(tool, dict)
        ]
        return SkillCreationResult(
            skill_id=installed_skill.id,
            skill_name=installed_skill.name,
            tools=tool_names,
            files_count=len(files.scripts) + 2,
            summary=f"Skill '{installed_skill.name}' 创建成功",
        )

    async def _analyze_requirement(self, description: str) -> SkillBlueprint:
        import asyncio

        messages = [
            SystemMessage(content=ANALYZE_SYSTEM_PROMPT),
            HumanMessage(content=description),
        ]
        structured = self._llm.with_structured_output(SkillBlueprint)

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                return await structured.ainvoke(messages)
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    delay = 2 * (2 ** attempt)
                    logger.warning(
                        "蓝图分析失败 (attempt %d/3), %ds 后重试: %s",
                        attempt + 1, delay, exc,
                    )
                    await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    async def _research(self, blueprint: SkillBlueprint) -> list[GitHubRepoInfo]:
        try:
            return await self._github.research_keywords(blueprint.search_keywords, top_n=3)
        except Exception as exc:
            logger.warning("GitHub 调研失败，降级为空结果: %s", exc)
            return []

    async def _generate_files(
        self,
        blueprint: SkillBlueprint,
        research_report: str,
    ) -> SkillGeneratedFiles:
        """调用 LLM 生成 Skill 文件集合。

        使用双路策略确保 LLM 同时遵循结构约束和内容约束：
        1. with_structured_output(SkillGeneratedFiles) 保证返回 JSON 结构
           （Field descriptions 携带 manifest/bundle/pip 等格式要求）
        2. GENERATE_SYSTEM_PROMPT 提供详细的格式说明和代码示例
        3. _validate_generated_files 做结构校验，不通过则重试
        """
        import asyncio

        tool_payload = [tool.model_dump() for tool in blueprint.tools]
        prompt = (
            "## 需求蓝图\n"
            f"- 名称: {blueprint.skill_name}\n"
            f"- 描述: {blueprint.description}\n"
            f"- 工具: {json.dumps(tool_payload, ensure_ascii=False)}\n"
            f"- 依赖: {json.dumps(blueprint.estimated_deps, ensure_ascii=False)}\n\n"
            f"{research_report}"
        )
        messages = [
            SystemMessage(content=GENERATE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        # 优先 with_structured_output（强制 JSON），失败时回退 ainvoke + JSON 解析
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                if attempt < 2:
                    # 前两次用 with_structured_output（tool call 强制 JSON）
                    structured = self._llm.with_structured_output(SkillGeneratedFiles)
                    files: SkillGeneratedFiles = await structured.ainvoke(messages)
                else:
                    # 最后一次回退：直接 ainvoke + JSON 解析
                    response = await self._llm.ainvoke(
                        messages,
                        response_format={"type": "json_object"},
                    )
                    text = response.content if hasattr(response, "content") else str(response)
                    parsed = _parse_llm_json(text)
                    files = SkillGeneratedFiles.model_validate(parsed)

                # 结构化校验：LLM 必须生成有效的可执行 Skill
                _validate_generated_files(files)

                return files
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    delay = 2 * (2 ** attempt)
                    logger.warning(
                        "Skill 文件生成失败 (attempt %d/3), %ds 后重试: %s",
                        attempt + 1, delay, exc,
                    )
                    await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    async def _validate_in_sandbox(
        self,
        files: SkillGeneratedFiles,
        sandbox: Sandbox,
    ) -> list[str]:
        """沙箱端到端验证：语法 → 部署 → 依赖安装 → --help → 烟雾测试。

        烟雾测试会用 manifest 中每个工具的参数构造最小 JSON 输入，
        实际执行脚本并检查是否能跑通（输出 JSON / 不崩溃）。
        """
        errors: list[str] = []

        # ---- 1. 语法检查 ---- #
        for script in files.scripts:
            if not script.path.endswith(".py"):
                errors.append(
                    f"脚本 {script.path} 不是 Python 文件，"
                    "entry.command 要求 bundle/*.py 格式"
                )
                continue
            try:
                ast.parse(script.content)
            except SyntaxError as exc:
                errors.append(f"语法错误 ({script.path}): {exc}")
        if errors:
            return errors

        # ---- 2. 部署到沙箱 ---- #
        session_id = "skill_creator_validate"
        skill_root = "/tmp/skill_creator_validate"

        mkdir_result = await sandbox.exec_command(
            session_id=session_id,
            exec_dir="/tmp",
            command=f"rm -rf {skill_root} && mkdir -p {skill_root}/bundle",
        )
        if not mkdir_result.success:
            errors.append(f"创建验证目录失败: {mkdir_result.message or ''}".strip())
            return errors

        for script in files.scripts:
            write_result = await sandbox.write_file(
                filepath=f"{skill_root}/{script.path}",
                content=script.content,
            )
            if not write_result.success:
                errors.append(
                    f"写入脚本失败 ({script.path}): {write_result.message or ''}".strip()
                )
                return errors

        # ---- 3. 依赖安装 ---- #
        if files.dependencies:
            deps_str = " ".join(shlex.quote(dep) for dep in files.dependencies)
            install_result = await sandbox.exec_command(
                session_id=session_id,
                exec_dir=skill_root,
                command=f"python -m pip install {deps_str}",
            )
            if not install_result.success:
                errors.append(f"依赖安装失败: {install_result.message or ''}".strip())
                return errors

        # ---- 4. --help 测试 ---- #
        for script in files.scripts:
            run_result = await sandbox.exec_command(
                session_id=session_id,
                exec_dir=skill_root,
                command=f"python {shlex.quote(script.path)} --help",
            )
            if not run_result.success:
                errors.append(f"脚本 --help 失败 ({script.path}): {run_result.message or ''}".strip())
                return errors

        # ---- 5. 烟雾测试：用最小输入实际运行每个工具 ---- #
        smoke_errors = await self._smoke_test_tools(files, sandbox, session_id, skill_root)
        errors.extend(smoke_errors)

        return errors

    async def _smoke_test_tools(
        self,
        files: SkillGeneratedFiles,
        sandbox: Sandbox,
        session_id: str,
        skill_root: str,
    ) -> list[str]:
        """对 manifest 中每个 tool 做实际运行烟雾测试。

        根据 tool 的 parameters 构造最小 JSON 输入，执行 entry.command，
        验证脚本能正常启动、处理输入并输出 JSON。
        """
        errors: list[str] = []
        manifest_tools = files.manifest.get("tools", [])
        if not manifest_tools:
            return errors

        for tool_def in manifest_tools:
            if not isinstance(tool_def, dict):
                continue
            tool_name = tool_def.get("name", "unknown")
            entry = tool_def.get("entry", {})
            command = entry.get("command", "") if isinstance(entry, dict) else ""
            if not command:
                continue

            # 构造最小测试输入
            sample_input = _build_sample_input(tool_def)
            sample_json = json.dumps(sample_input, ensure_ascii=False)

            # 执行: python bundle/tool.py '{"param": "test"}'
            full_cmd = f"{command} {shlex.quote(sample_json)}"
            run_result = await sandbox.exec_command(
                session_id=session_id,
                exec_dir=skill_root,
                command=full_cmd,
            )

            output = (run_result.message or "").strip()

            if not run_result.success:
                errors.append(
                    f"工具 {tool_name} 烟雾测试失败 "
                    f"(cmd: {command}):\n{output[:500]}"
                )
                continue

            # 检查 stdout 是否包含 JSON 输出
            if not output:
                errors.append(
                    f"工具 {tool_name} 烟雾测试无输出 "
                    f"(预期输出 JSON 到 stdout)"
                )
                continue

            # 尝试解析 JSON（允许输出中有前缀日志，取最后一行）
            json_found = False
            for line in reversed(output.splitlines()):
                line = line.strip()
                if line.startswith(("{", "[")):
                    try:
                        json.loads(line)
                        json_found = True
                        break
                    except json.JSONDecodeError:
                        pass
            if not json_found:
                errors.append(
                    f"工具 {tool_name} 烟雾测试输出非 JSON:\n{output[:300]}"
                )

        return errors

    async def _fix_files(
        self,
        files: SkillGeneratedFiles,
        errors: list[str],
        blueprint: SkillBlueprint,
        research_report: str,
    ) -> SkillGeneratedFiles:
        del files
        fix_patch = (
            "\n\n## 验证错误\n"
            + "\n".join(f"- {item}" for item in errors)
            + "\n请修复后返回完整 JSON。"
        )
        return await self._generate_files(blueprint, research_report + fix_patch)

    async def _install(self, files: SkillGeneratedFiles, installed_by: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir)

            (skill_dir / "SKILL.md").write_text(files.skill_md, encoding="utf-8")
            (skill_dir / "manifest.json").write_text(
                json.dumps(files.manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            for script in files.scripts:
                script_path = skill_dir / script.path
                script_path.parent.mkdir(parents=True, exist_ok=True)
                script_path.write_text(script.content, encoding="utf-8")

            return await self._skill_service.install_skill(
                source_type=SkillSourceType.LOCAL,
                source_ref=f"local:{skill_dir.as_posix()}",
                manifest=files.manifest,
                skill_md=files.skill_md,
                installed_by=installed_by,
            )

