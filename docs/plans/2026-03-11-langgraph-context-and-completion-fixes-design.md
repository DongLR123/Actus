# LangGraph 迁移后上下文丢失与任务完成缺陷修复设计

日期: 2026-03-11
状态: 已确认

## 背景

LangGraph 迁移（commit 014856c ~ 9001365）将 PlannerReActFlow 从旧版 while 循环 + BaseAgent 架构重写为 LangGraph StateGraph。迁移过程中丢失了若干关键行为，导致三类用户可感知的问题：

1. **LLM 默认语言为英文** — 系统 prompt 声明默认中文，但实际传给 LLM 的 language 参数始终为 "en"
2. **上下文丢失** — 多轮对话中 planner 不知道之前聊了什么，executor 在非首轮时跳过系统 prompt
3. **任务"完成"但无输出** — 步骤无条件标记成功，summarizer 不生成最终总结

## 根因分析

### Bug 1: Language 默认值 "en"

- `Message` 模型没有 `language` 字段
- `planner_react.py:257`: `getattr(message, "language", "en")` 永远返回 "en"
- `main_graph.py` 中所有 `state.get("language", "en")` fallback 也是 "en"

### Bug 2: Planner 无历史上下文

- 旧版: `BaseAgent._build_effective_system_prompt()` 为 planner 和 react 都注入 conversation_summaries + context_anchor
- 新版: `planner_node` 只收到 `PLANNER_SYSTEM_PROMPT` + `CREATE_PLAN_PROMPT(message=...)`，没有 summaries

### Bug 3: Executor messages 恢复 vs 新轮次混淆

- 旧版: react 有独立持久 memory，`_ensure_system_message()` 确保首条是最新 system prompt
- 新版: `executor_node` 判断 `saved_messages` 非空就走 "恢复" 分支（注入 "用户已完成接管" 消息 + 跳过 system prompt），但非恢复的第二轮对话也会命中

### Bug 4: Step 无条件标记成功

- `executor_node` lines 197-199: 无条件 `step.status = COMPLETED; step.success = True`
- 旧版: react agent 解析 LLM 返回的 JSON 中的 `success` 字段

### Bug 5: Summarizer 不生成总结

- `summarizer_node` 只发 `PlanEvent(COMPLETED)` + `DoneEvent()`
- `SUMMARIZE_PROMPT` 已定义但未使用
- 旧版: `react.summarize()` 调用 LLM 生成最终总结

## 设计决策

| 问题 | 决策 | 理由 |
|------|------|------|
| Planner 上下文来源 | 仅注入 conversation_summaries（方案 A） | Planner 负责意图理解和步骤拆分，不需要工具调用细节；summaries 已足够 |
| Summaries 注入位置 | 追加到 planner 的 system prompt（非 user prompt） | 语义正确（背景知识而非用户指令）；与 executor 注入方式对称；不改 prompt 模板 |
| Executor messages 处理 | 还原旧版持久 memory 模式（方案 C） | 同一 plan 多步骤需要共享工具调用历史；与旧版行为一致 |
| Language 默认值 | 改为 "zh" | 与系统 prompt 中"默认工作语言：中文"一致；不改 Message 模型 |

## 详细设计

### 1. Language 默认值修复

**文件**: `planner_react.py`, `main_graph.py`

将所有 `"en"` fallback 改为 `"zh"`:
- `planner_react.py:257`: `getattr(message, "language", "zh")`
- `main_graph.py` planner_node: `parsed.get("language", state.get("language", "zh"))`
- `main_graph.py` executor_node: `state.get("language", "zh")`

### 2. Planner 上下文注入

**文件**: `main_graph.py` planner_node

在 `planner_node` 中动态构建 system_content，将 conversation_summaries 追加到 PLANNER_SYSTEM_PROMPT 之后：

```python
system_content = PLANNER_SYSTEM_PROMPT
conversation_summaries = state.get("conversation_summaries") or []
if conversation_summaries:
    system_content += "\n\n## 历史对话摘要\n" + "\n\n".join(conversation_summaries)

response = await planner_llm.invoke(
    messages=[
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ],
    ...
)
```

不改动 `CREATE_PLAN_PROMPT` 模板。

### 3. Executor 持久 Memory 还原

**文件**: `state.py`, `main_graph.py`, `planner_react.py`

#### 3a. State 增加 is_resuming 字段

`MainGraphState` 新增 `is_resuming: bool`，由 `planner_react.py` 在 input_state 中设置。

#### 3b. executor_node 三分支 messages 逻辑

```
if is_resuming and saved_messages:
    # 中断恢复：保留消息 + 追加恢复提示
    initial_messages = saved_messages + [恢复 user msg]
elif saved_messages:
    # 非首步/有历史：更新 system prompt，追加新 execution prompt
    saved_messages[0]["content"] = system_content
    initial_messages = saved_messages + [execution user msg]
else:
    # 首步/无历史：干净的 system + execution prompt
    initial_messages = [system msg, execution user msg]
```

#### 3c. Memory 生命周期

1. `planner_react.py` invoke 前从 DB 加载 memory → 传入 `input_state["messages"]`
2. executor 每步执行后 messages 在 state 中累积
3. updater → executor 循环中 messages 自动在 state 中保持
4. 任务完成后 `planner_react.py` 将 final messages compact 后存回 DB（已有逻辑）

### 4. Step 成功状态检测

**文件**: `main_graph.py` executor_node

替换无条件成功为 JSON 解析检测：

```python
react_messages = react_final.get("messages", [])
step_success = True
summary = ""
for msg in reversed(react_messages):
    if msg.get("role") == "assistant" and msg.get("content"):
        summary = msg["content"][:500]
        try:
            result_json = json.loads(msg["content"])
            step_success = result_json.get("success", True)
        except (json.JSONDecodeError, TypeError):
            pass  # 非 JSON 回复视为成功
        break

step.status = ExecutionStatus.COMPLETED
step.success = step_success
```

`step.success = False` 的步骤仍标记为 COMPLETED（已执行完毕），updater 可据此重新规划后续步骤。

### 5. Summarizer 生成最终总结

**文件**: `main_graph.py` summarizer_node

调用 `summary_llm` + `SUMMARIZE_PROMPT`，将结果作为 `MessageEvent` 发出：

```python
async def summarizer_node(state: MainGraphState) -> dict:
    plan = state["plan"]
    events = []

    if plan:
        plan.status = ExecutionStatus.COMPLETED
        events.append(PlanEvent(plan=plan, status=PlanEventStatus.COMPLETED))

    react_messages = state.get("messages", [])
    if react_messages:
        response = await summary_llm.invoke(
            messages=react_messages + [
                {"role": "user", "content": SUMMARIZE_PROMPT}
            ],
        )
        summary_content = response.get("content", "")
        if summary_content:
            events.append(MessageEvent(role="assistant", message=summary_content))

    events.append(DoneEvent())
    return {"flow_status": "completed", "events": events}
```

## 涉及文件清单

| 文件 | 改动类型 |
|------|---------|
| `api/app/domain/services/flows/planner_react.py` | 修改: language 默认值, is_resuming 字段传入 |
| `api/app/domain/services/graphs/main_graph.py` | 修改: planner 注入 summaries, executor 三分支, step 成功检测, summarizer 生成总结 |
| `api/app/domain/services/graphs/state.py` | 修改: MainGraphState 增加 is_resuming |
| `api/app/domain/services/prompts/planner.py` | 不改动 |
| `api/app/domain/services/prompts/react.py` | 不改动（SUMMARIZE_PROMPT 已存在） |

## 测试策略

- 单元测试覆盖 executor_node 三分支（无历史 / 有历史 / 中断恢复）
- 单元测试覆盖 step success 解析（JSON 成功 / JSON 失败 / 非 JSON）
- 集成测试覆盖多轮对话场景（验证 planner 能看到 summaries）
