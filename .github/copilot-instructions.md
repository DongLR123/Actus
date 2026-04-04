# Actus Copilot Instructions

## 项目概述

这是一个 **开源 AI Agent 平台**，基于 MCP (Model Context Protocol) 工具协议与 A2A (Agent-to-Agent) 智能体间通信协议，构建具有工具调用能力的 ReAct Agent。支持 **Skill 生态系统**，通过 SKILL.md 定义与安装扩展能力，基于文件系统存储和管理。

## 架构模式

### 整体架构
项目由三个子服务组成：
- `api/` — 后端服务 (FastAPI)，采用 **整洁架构 (Clean Architecture) + DDD**
- `ui/` — 前端 (Next.js 16 + React 19)
- `sandbox/` — 沙箱服务（Docker 隔离代码执行）

### ReAct Agent 实现模式
项目采用 **Planner + ReAct** 双阶段模式：
1. **Planner 阶段**：用户输入 → LLM 生成执行计划（步骤列表）
2. **ReAct 阶段**：逐步执行，每步循环：LLM 推理 → 工具调用 → 结果整合 → 递归/输出
3. **上下文治理**：执行过程中通过 TokenEstimator + ContextAssembler + GradualCompactor 自动管理上下文窗口

### Skill 生态系统
Skill 是独立的扩展生态层，通过 SKILL.md 定义，支持三类运行时：
- **native** — 通过 Sandbox 执行命令
- **mcp** — 委托 MCP 服务器工具
- **a2a** — 委托远程 Agent

Skill 存储在 **文件系统**（默认 `/app/data/skills`），每个 Skill 为独立目录：
```
/app/data/skills/{skill_id}/
├── meta.json          # 元信息
├── manifest.json      # 工具定义、策略配置
├── SKILL.md           # 说明文档（含 frontmatter）
├── bundle_index.json  # 文件索引
└── bundle/            # 源文件
```

Skill 安装支持 GitHub、本地目录和 SKILL.md 格式。基于 Embedding 的语义选择在规划阶段自动匹配最相关的 Skill。

Skill 相关核心代码路径：
- 领域模型：`app/domain/models/skill.py`
- 仓库接口：`app/domain/repositories/skill_repository.py`
- 仓库实现：`app/infrastructure/repositories/file_skill_repository.py`（当前使用）
- 应用服务：`app/application/services/skill_service.py`、`skill_source_loader.py`、`skill_selector.py`、`skill_index_service.py`
- SKILL.md 解析/导出：`app/domain/services/skill_md_parser.py`、`skill_md_exporter.py`
- Embedding 索引：`app/infrastructure/external/embedding/skill_embedding_index.py`
- 动态工具转换：`app/domain/services/tools/langchain_dynamic_skill_tools.py`
- 统一工具层：`app/domain/services/tools/skill.py`（SkillTool）
- API 路由：`app/interfaces/endpoints/skill_v2_routes.py`（v2，当前活跃）

### MCP 渐进式发现
`app/domain/services/tools/langchain_mcp_discovery.py` 提供两个工具：
- `list_mcp_tools(server_name?)` — 列出可用 MCP 工具（仅 metadata）
- `get_mcp_tool(tool_name)` — 获取完整 schema 并激活到下一轮执行

### 多模态文件理解
`app/infrastructure/external/file_processors/` 按 MIME 类型路由：
- **audio** — OpenAI Whisper API 或 sandbox faster-whisper 转录
- **pdf** — 原生 PDF blocks 或 pymupdf4llm 提取
- **image** — Base64 编解码、MIME 校验、5MB 限制
- **video** — 关键帧提取 + 视觉模型分析 + 音频转录

### 上下文溢出治理
`app/domain/services/graphs/` 中三层递进保护：
- **TokenEstimator** — 混合 token 估算（图片 2000、PDF ~2000/页、CJK 1.5/字）
- **ContextAssembler** — 同步三阶段裁剪
- **GradualCompactor** — 异步两级压缩（85% LLM 摘要 / 95% 硬截断）

### 工具定义约定
使用 **Pydantic BaseModel** 定义工具输入参数 schema：

```python
from pydantic import BaseModel, Field

class ToolInput(BaseModel):
    param: str = Field(..., description="参数描述")

# 转换为 OpenAI tools 格式
tools = [{
    "type": "function",
    "function": {
        "name": tool_func.__name__,
        "description": tool_func.__doc__,  # 使用函数 docstring
        "parameters": ToolInput.model_json_schema(),
    }
}]
```

## 环境配置

- 使用 `.env` 文件管理环境变量（参考 `.env.example`）
- 运行时配置使用 `api/config.yaml`（参考 `api/config.yaml.example`）
- Skill 存储路径通过 `SKILLS_ROOT_DIR` 环境变量配置（默认 `/app/data/skills`）
- 敏感信息（密钥、密码）通过环境变量注入，**不硬编码**

## 常用模型标识

| Provider | Model ID | 用途 |
|----------|----------|------|
| DeepSeek | `deepseek-chat` | 通用对话 |
| DeepSeek | `deepseek-reasoner` | 推理增强 |
| Kimi | `kimi-k2-0905-preview` | 通用对话 |

## 流式响应处理

处理流式 tool_calls 时需要按 `index` 聚合分片：
```python
tool_calls_obj: dict[int, ChoiceDeltaToolCall] = {}
for chunk in response:
    if chunk_tool_calls := chunk.choices[0].delta.tool_calls:
        for tc in chunk_tool_calls:
            if tc.index not in tool_calls_obj:
                tool_calls_obj[tc.index] = tc
            else:
                tool_calls_obj[tc.index].function.arguments += tc.function.arguments
```

## 开发命令

```bash
# 安装依赖（使用 uv）
uv sync

# 后端开发
cd api && bash dev.sh

# 前端开发
cd ui && npm run dev

# 运行测试
pytest api/tests/          # 后端测试
cd ui && npm run test      # 前端测试

# Docker 部署
cp .env.example .env       # 配置环境变量
cp api/config.yaml.example api/config.yaml  # 配置运行时参数
docker compose --env-file .env up -d --build
```

## 注意事项

- 工具调用后必须将结果以 `role: "tool"` 消息格式追加到 messages
- `tool_call_id` 必须与对应的 tool_call 匹配
- 递归调用 `process_query()` 时不要重复添加用户消息
- Domain 层禁止导入 FastAPI/SQLAlchemy，仅允许 langchain-core 和 langgraph
- 后端全异步：所有 I/O 操作使用 `async/await`
- 前端禁止 `any` 类型
- 消息清洗器 (`message_sanitizer.py`) 在 LLM 调用前自动过滤不合规的多模态内容
