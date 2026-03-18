"""Skill Creator 领域模型。"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class ToolParamDef(BaseModel):
    """工具参数定义。"""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = False


class ToolDef(BaseModel):
    """工具定义。"""

    name: str
    description: str
    parameters: list[ToolParamDef] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_parameters(cls, data: Any) -> Any:
        """兼容 parameters 为对象字典或 JSON Schema 格式的场景。"""
        if not isinstance(data, dict):
            return data

        raw_parameters = data.get("parameters")
        if raw_parameters is None:
            normalized = dict(data)
            normalized["parameters"] = []
            return normalized

        if not isinstance(raw_parameters, dict):
            return data

        # 检测 JSON Schema 格式: {"type": "object", "properties": {...}, ...}
        if "properties" in raw_parameters and isinstance(
            raw_parameters.get("properties"), dict
        ):
            properties = raw_parameters["properties"]
            schema_required = raw_parameters.get("required", [])
            required_names = (
                {str(item) for item in schema_required if isinstance(item, str)}
                if isinstance(schema_required, list)
                else set()
            )
            normalized_parameters: list[dict[str, Any]] = []
            for name, spec in properties.items():
                param_name = str(name).strip()
                if not param_name:
                    continue
                if isinstance(spec, dict):
                    normalized_parameters.append(
                        {
                            "name": param_name,
                            "type": str(spec.get("type") or "string"),
                            "description": str(spec.get("description") or ""),
                            "required": param_name in required_names,
                        }
                    )
                else:
                    normalized_parameters.append(
                        {
                            "name": param_name,
                            "type": "string",
                            "description": "",
                            "required": param_name in required_names,
                        }
                    )
            normalized = dict(data)
            normalized["parameters"] = normalized_parameters
            return normalized

        # 原有逻辑：parameters 为扁平对象字典 {"param_name": {type, description}, ...}
        required_raw = data.get("required")
        required_names = (
            {str(item) for item in required_raw if isinstance(item, str)}
            if isinstance(required_raw, list)
            else set()
        )

        normalized_parameters = []
        for name, spec in raw_parameters.items():
            param_name = str(name).strip()
            if not param_name:
                continue
            if isinstance(spec, dict):
                normalized_parameters.append(
                    {
                        "name": param_name,
                        "type": str(spec.get("type") or "string"),
                        "description": str(spec.get("description") or ""),
                        "required": bool(
                            spec.get("required", param_name in required_names)
                        ),
                    }
                )
            else:
                normalized_parameters.append(
                    {
                        "name": param_name,
                        "type": "string",
                        "description": "",
                        "required": param_name in required_names,
                    }
                )

        normalized = dict(data)
        normalized["parameters"] = normalized_parameters
        return normalized


class SkillBlueprint(BaseModel):
    """由 LLM 从需求中提取的结构化蓝图。"""

    skill_name: str
    description: str
    tools: list[ToolDef] = Field(default_factory=list)
    search_keywords: list[str] = Field(default_factory=list)
    estimated_deps: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_llm_output(cls, data: Any) -> Any:
        """容错处理 LLM 常见的 JSON 结构偏差。"""
        if not isinstance(data, dict):
            return data

        # 1. 解包嵌套：{"skill": {...}}, {"blueprint": {...}}, {"result": {...}}
        unwrap_keys = ("skill", "blueprint", "result", "data")
        for key in unwrap_keys:
            if key in data and isinstance(data[key], dict) and "skill_name" not in data:
                inner = data[key]
                if "skill_name" in inner or "name" in inner:
                    data = inner
                    break

        data = dict(data)

        # 2. 字段名映射：name → skill_name
        if "skill_name" not in data and "name" in data:
            data["skill_name"] = data.pop("name")

        # 3. 兼容 LLM 返回嵌套数组格式的 search_keywords
        raw_keywords = data.get("search_keywords")
        if isinstance(raw_keywords, list):
            flattened: list[str] = []
            for item in raw_keywords:
                if isinstance(item, list):
                    flattened.extend(str(kw) for kw in item if kw)
                elif isinstance(item, str):
                    flattened.append(item)
            if flattened != raw_keywords:
                data["search_keywords"] = flattened

        return data

    @computed_field
    @property
    def normalized_slug(self) -> str:
        slug = self.skill_name.strip().lower()
        slug = re.sub(r"[^a-z0-9]+", "-", slug)
        slug = re.sub(r"-{2,}", "-", slug).strip("-")
        return slug or "skill"


class ScriptFile(BaseModel):
    """生成的脚本文件。"""

    path: str = Field(
        description='脚本路径，必须以 "bundle/" 开头，如 "bundle/my_tool.py"',
    )
    content: str = Field(
        description="完整的 Python 脚本代码。脚本必须：1) 接受 sys.argv[1] 作为 JSON 输入；"
        "2) 输出 JSON 到 stdout；3) 支持 --help 参数",
    )


class SkillGeneratedFiles(BaseModel):
    """生成阶段输出的 Skill 文件集合。"""

    model_config = ConfigDict(extra="ignore")

    skill_md: str = Field(
        default="",
        description="完整的 SKILL.md 内容，必须包含 YAML frontmatter（--- 分隔）"
        "和 Markdown 使用说明",
    )
    manifest: dict[str, Any] = Field(
        default_factory=dict,
        description='manifest.json 对象，必须包含 name, runtime_type("native"), '
        "tools 数组。每个 tool 必须有 name, description, parameters(object 格式), "
        'required, entry(含 command: "python bundle/<name>.py")',
    )
    scripts: list[ScriptFile] = Field(
        default_factory=list,
        description="可执行 Python 脚本列表，每个脚本的 path 必须以 bundle/ 开头",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="pip 包名列表（如 requests, beautifulsoup4），不要写中文描述",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_generated_files(cls, data: Any) -> Any:
        """容错处理 LLM 常见的输出偏差。

        - 解包嵌套：{"result": {...}}, {"generated_files": {...}}
        - manifest 缺失时尝试从 skill_md 中解析 YAML frontmatter
        """
        if not isinstance(data, dict):
            return data

        # 解包嵌套
        unwrap_keys = ("result", "generated_files", "data", "output")
        for key in unwrap_keys:
            if key in data and isinstance(data[key], dict):
                inner = data[key]
                if "skill_md" in inner or "manifest" in inner or "scripts" in inner:
                    data = inner
                    break

        data = dict(data)

        # scripts: LLM 可能返回 dict {"tool_name": "code..."} 而不是 list [{path, content}]
        raw_scripts = data.get("scripts")
        if isinstance(raw_scripts, dict):
            normalized_scripts = []
            for name, content in raw_scripts.items():
                if isinstance(content, str):
                    path = name if name.startswith("bundle/") else f"bundle/{name}"
                    if not path.endswith(".py"):
                        path += ".py"
                    normalized_scripts.append({"path": path, "content": content})
                elif isinstance(content, dict):
                    # {"tool_name": {"path": ..., "content": ...}}
                    path = content.get("path", f"bundle/{name}.py")
                    code = content.get("content", content.get("code", ""))
                    if not path.startswith("bundle/"):
                        path = f"bundle/{path}"
                    normalized_scripts.append({"path": path, "content": code})
            data["scripts"] = normalized_scripts

        # dependencies: LLM 可能返回 [{name, description}] 或 dict 而不是 ["pkg_name"]
        raw_deps = data.get("dependencies")
        if isinstance(raw_deps, dict):
            # LLM 返回了 {"external_capabilities": [...], "pip_packages": [...]} 等结构
            # 尝试提取所有字符串值，过滤中文描述
            extracted: list[str] = []
            for _key, val in raw_deps.items():
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, str):
                            extracted.append(item)
                elif isinstance(val, str):
                    extracted.append(val)
            data["dependencies"] = [
                d for d in extracted
                if not any('\u4e00' <= c <= '\u9fff' for c in d)
            ]
        elif isinstance(raw_deps, list):
            normalized_deps = []
            for dep in raw_deps:
                if isinstance(dep, str):
                    normalized_deps.append(dep)
                elif isinstance(dep, dict):
                    name = dep.get("name", "")
                    # 跳过中文描述性 "依赖"（不是 pip 包名）
                    if name and not any('\u4e00' <= c <= '\u9fff' for c in name):
                        normalized_deps.append(name)
            data["dependencies"] = normalized_deps

        # manifest 缺失时尝试从 skill_md 解析
        if not data.get("manifest") and data.get("skill_md"):
            data["manifest"] = _extract_manifest_from_skill_md(data["skill_md"])

        # 自动补全 manifest 常见缺失字段（生成的 Skill 都是 native 类型）
        manifest = data.get("manifest")
        if isinstance(manifest, dict):
            if "runtime_type" not in manifest:
                manifest["runtime_type"] = "native"
            for tool in manifest.get("tools", []):
                if isinstance(tool, dict):
                    entry = tool.get("entry")
                    if isinstance(entry, dict) and "command" not in entry:
                        tool_name = tool.get("name", "tool")
                        entry["command"] = f"python bundle/{tool_name}.py"
                    elif entry is None:
                        tool_name = tool.get("name", "tool")
                        tool["entry"] = {"command": f"python bundle/{tool_name}.py"}

        return data


def _extract_manifest_from_skill_md(skill_md: str) -> dict[str, Any]:
    """从 SKILL.md 的 YAML frontmatter 中提取 manifest。

    SKILL.md 格式: ---\\n<yaml>\\n---\\n<markdown>
    """
    import re
    import yaml  # type: ignore[import-untyped]

    match = re.match(r"^---\s*\n(.*?)\n---", skill_md, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = yaml.safe_load(match.group(1))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


class GitHubRepoInfo(BaseModel):
    """GitHub 仓库摘要。"""

    name: str
    full_name: str
    description: str = ""
    stars: int = 0
    url: str
    readme_summary: str = ""
    install_command: str = ""


class SkillCreationProgress(BaseModel):
    """Skill 创建过程中的进度事件。"""

    step: Literal["analyzing", "researching", "generating", "validating", "installing"]
    message: str
    detail: str | None = None
    references: list[GitHubRepoInfo] | None = None


class SkillCreationResult(BaseModel):
    """Skill 创建完成结果。"""

    skill_id: str
    skill_name: str
    tools: list[str]
    files_count: int
    summary: str
