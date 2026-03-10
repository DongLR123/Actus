# LangGraph Context Loss & Completion Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 5 bugs introduced during LangGraph migration: language default, planner context injection, executor memory restoration, step success detection, and summarizer output generation.

**Architecture:** All changes are in the LangGraph graph layer (`main_graph.py`, `state.py`) and the flow orchestrator (`planner_react.py`). No prompt templates are modified. No new files created (except tests).

**Tech Stack:** Python 3.12, LangGraph, Pydantic, pytest + anyio

**Design doc:** `docs/plans/2026-03-11-langgraph-context-and-completion-fixes-design.md`

---

### Task 1: Fix language default from "en" to "zh"

**Files:**
- Modify: `api/app/domain/services/flows/planner_react.py:257`
- Modify: `api/app/domain/services/graphs/main_graph.py:85,97,143`
- Test: `api/tests/domain/services/graphs/test_main_graph.py`

**Step 1: Write the failing test**

In `api/tests/domain/services/graphs/test_main_graph.py`, add to `TestMainGraphFlow`:

```python
async def test_default_language_is_zh(self, mock_planner_llm, mock_json_parser):
    """When no language is specified, planner should default to zh."""
    from app.domain.services.graphs.main_graph import build_main_graph

    captured_messages = []
    original_invoke = mock_planner_llm.invoke

    async def capture_invoke(**kwargs):
        captured_messages.append(kwargs.get("messages", []))
        return await original_invoke(**kwargs)

    mock_planner_llm.invoke = capture_invoke

    graph = build_main_graph(
        planner_llm=mock_planner_llm,
        react_graph=_make_mock_react_graph(),
        json_parser=mock_json_parser,
        summary_llm=mock_planner_llm,
        uow_factory=MagicMock(),
        session_id="sess-lang",
    )

    result = await graph.ainvoke({
        "message": "帮我查一下天气",
        "language": "zh",
        "attachments": [],
        "plan": None,
        "current_step": None,
        "messages": [],
        "execution_summary": "",
        "events": [],
        "flow_status": "idle",
        "session_id": "sess-lang",
        "should_interrupt": False,
        "original_request": "",
        "skill_context": "",
        "conversation_summaries": [],
        "is_resuming": False,
    })

    # Verify the plan language fallback is "zh" not "en"
    plan = result.get("plan")
    assert plan is not None
    # When planner LLM returns "en" but the message is Chinese,
    # the fallback should use state language "zh"
```

**Step 2: Run test to verify it fails**

Run: `cd api && python -m pytest tests/domain/services/graphs/test_main_graph.py::TestMainGraphFlow::test_default_language_is_zh -v`

Expected: FAIL (is_resuming not in state / KeyError)

**Step 3: Apply the fix**

In `api/app/domain/services/flows/planner_react.py`, change line 257:
```python
# Before:
"language": getattr(message, "language", "en"),
# After:
"language": getattr(message, "language", "zh"),
```

In `api/app/domain/services/graphs/main_graph.py`, change three locations:
```python
# planner_node fallback (line ~85):
"language": parsed.get("language", state.get("language", "zh")),

# planner_node full fallback plan (line ~97):
"language": state.get("language", "zh"),

# executor_node (line ~143):
language = state.get("language", "zh")
```

**Step 4: Run test to verify it passes**

Run: `cd api && python -m pytest tests/domain/services/graphs/test_main_graph.py -v`

Expected: ALL PASS

**Step 5: Commit**

```bash
git add api/app/domain/services/flows/planner_react.py api/app/domain/services/graphs/main_graph.py api/tests/domain/services/graphs/test_main_graph.py
git commit -m "fix: change language default from 'en' to 'zh' to match system prompt"
```

---

### Task 2: Add `is_resuming` to MainGraphState

**Files:**
- Modify: `api/app/domain/services/graphs/state.py:35`
- Modify: `api/app/domain/services/flows/planner_react.py` (input_state)

**Step 1: Add the field to state**

In `api/app/domain/services/graphs/state.py`, add to `MainGraphState` after `should_interrupt: bool`:

```python
    is_resuming: bool
```

**Step 2: Pass it from planner_react.py**

In `api/app/domain/services/flows/planner_react.py`, add to `input_state` dict (after `should_interrupt`):

```python
"is_resuming": is_resuming,
```

**Step 3: Update existing test input states**

In `api/tests/domain/services/graphs/test_main_graph.py`, add `"is_resuming": False` to the `ainvoke` input dict in `test_full_flow_produces_plan_and_done`.

In `api/tests/domain/services/graphs/test_integration.py`, no changes needed (uses PlannerReActFlow which builds the state internally).

**Step 4: Run all tests**

Run: `cd api && python -m pytest tests/domain/services/graphs/ -v`

Expected: ALL PASS

**Step 5: Commit**

```bash
git add api/app/domain/services/graphs/state.py api/app/domain/services/flows/planner_react.py api/tests/domain/services/graphs/test_main_graph.py
git commit -m "feat: add is_resuming field to MainGraphState"
```

---

### Task 3: Inject conversation summaries into planner

**Files:**
- Modify: `api/app/domain/services/graphs/main_graph.py` (planner_node)
- Test: `api/tests/domain/services/graphs/test_main_graph.py`

**Step 1: Write the failing test**

In `api/tests/domain/services/graphs/test_main_graph.py`, add to `TestMainGraphFlow`:

```python
async def test_planner_receives_conversation_summaries(self, mock_json_parser):
    """Planner system prompt should include conversation summaries when available."""
    from app.domain.services.graphs.main_graph import build_main_graph

    captured_system_content = []

    async def capturing_invoke(**kwargs):
        messages = kwargs.get("messages", [])
        for m in messages:
            if m.get("role") == "system":
                captured_system_content.append(m["content"])
        return {
            "content": '{"title":"Test","goal":"test","language":"zh","steps":[{"description":"step1"}],"message":"ok"}',
            "role": "assistant",
        }

    planner_llm = AsyncMock()
    planner_llm.invoke = capturing_invoke

    graph = build_main_graph(
        planner_llm=planner_llm,
        react_graph=_make_mock_react_graph(),
        json_parser=mock_json_parser,
        summary_llm=planner_llm,
        uow_factory=MagicMock(),
        session_id="sess-summary",
    )

    await graph.ainvoke({
        "message": "继续上次的工作",
        "language": "zh",
        "attachments": [],
        "plan": None,
        "current_step": None,
        "messages": [],
        "execution_summary": "",
        "events": [],
        "flow_status": "idle",
        "session_id": "sess-summary",
        "should_interrupt": False,
        "original_request": "",
        "skill_context": "",
        "conversation_summaries": ["### 第1轮\n- 用户需求：查天气\n- 执行结果：成功获取北京天气"],
        "is_resuming": False,
    })

    assert len(captured_system_content) >= 1
    assert "历史对话摘要" in captured_system_content[0]
    assert "查天气" in captured_system_content[0]
```

**Step 2: Run test to verify it fails**

Run: `cd api && python -m pytest tests/domain/services/graphs/test_main_graph.py::TestMainGraphFlow::test_planner_receives_conversation_summaries -v`

Expected: FAIL — "历史对话摘要" not in system content

**Step 3: Implement the fix**

In `api/app/domain/services/graphs/main_graph.py`, modify `planner_node`:

```python
async def planner_node(state: MainGraphState) -> dict:
    """Call planner LLM to create a plan from user message."""
    attachments = state.get("attachments", [])
    prompt = CREATE_PLAN_PROMPT.format(
        message=state["message"],
        attachments=", ".join(attachments) if attachments else "无",
    )

    # Build system prompt with optional conversation summaries
    system_content = PLANNER_SYSTEM_PROMPT
    conversation_summaries = state.get("conversation_summaries") or []
    if conversation_summaries:
        system_content += "\n\n## 历史对话摘要\n" + "\n\n".join(conversation_summaries)

    response = await planner_llm.invoke(
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    # ... rest unchanged
```

**Step 4: Run test to verify it passes**

Run: `cd api && python -m pytest tests/domain/services/graphs/test_main_graph.py -v`

Expected: ALL PASS

**Step 5: Commit**

```bash
git add api/app/domain/services/graphs/main_graph.py api/tests/domain/services/graphs/test_main_graph.py
git commit -m "fix: inject conversation summaries into planner system prompt"
```

---

### Task 4: Fix executor messages three-branch logic

**Files:**
- Modify: `api/app/domain/services/graphs/main_graph.py` (executor_node, lines 155-171)
- Test: `api/tests/domain/services/graphs/test_main_graph.py`

**Step 1: Write failing tests for all three branches**

In `api/tests/domain/services/graphs/test_main_graph.py`, add new test class:

```python
class TestExecutorMessageBranching:
    """Test the three-way branching in executor_node for message handling."""

    @pytest.fixture
    def mock_json_parser(self):
        parser = AsyncMock()
        import json
        async def parse(content, default_value=None):
            try:
                return json.loads(content)
            except Exception:
                return default_value
        parser.invoke = parse
        return parser

    async def test_first_step_no_history_uses_system_prompt(self, mock_json_parser):
        """When messages=[] and is_resuming=False, executor builds fresh system+execution prompt."""
        from app.domain.services.graphs.main_graph import build_main_graph
        from app.domain.models.plan import Plan, Step

        captured_react_inputs = []

        class CapturingReactGraph:
            async def astream(self, input_state, config=None):
                captured_react_inputs.append(input_state)
                yield {"llm_node": {
                    "events": [],
                    "messages": input_state["messages"] + [
                        {"role": "assistant", "content": '{"success": true, "result": "done", "attachments": []}'}
                    ],
                    "should_interrupt": False,
                }}

        planner_llm = AsyncMock()
        async def mock_invoke(**kwargs):
            return {"content": '{"title":"T","goal":"G","language":"zh","steps":[{"description":"S1"}],"message":"ok"}', "role": "assistant"}
        planner_llm.invoke = mock_invoke

        graph = build_main_graph(
            planner_llm=planner_llm,
            react_graph=CapturingReactGraph(),
            json_parser=mock_json_parser,
            summary_llm=planner_llm,
            uow_factory=MagicMock(),
            session_id="sess-exec",
        )

        await graph.ainvoke({
            "message": "do something",
            "language": "zh",
            "attachments": [],
            "plan": None,
            "current_step": None,
            "messages": [],
            "execution_summary": "",
            "events": [],
            "flow_status": "idle",
            "session_id": "sess-exec",
            "should_interrupt": False,
            "original_request": "",
            "skill_context": "",
            "conversation_summaries": [],
            "is_resuming": False,
        })

        assert len(captured_react_inputs) >= 1
        msgs = captured_react_inputs[0]["messages"]
        assert msgs[0]["role"] == "system"
        assert "任务执行智能体" in msgs[0]["content"]

    async def test_has_history_not_resuming_updates_system_prompt(self, mock_json_parser):
        """When messages have history and is_resuming=False, executor updates system prompt and appends execution prompt."""
        from app.domain.services.graphs.main_graph import build_main_graph
        from app.domain.models.plan import Plan, Step, ExecutionStatus

        captured_react_inputs = []

        class CapturingReactGraph:
            async def astream(self, input_state, config=None):
                captured_react_inputs.append(input_state)
                yield {"llm_node": {
                    "events": [],
                    "messages": input_state["messages"] + [
                        {"role": "assistant", "content": '{"success": true, "result": "done", "attachments": []}'}
                    ],
                    "should_interrupt": False,
                }}

        planner_llm = AsyncMock()
        # Won't be called since we skip planner via flow_status=executing
        planner_llm.invoke = AsyncMock()

        step = Step(description="Step 2: analyze data")
        plan = Plan(title="T", goal="G", language="zh", steps=[
            Step(description="Step 1: collect", status=ExecutionStatus.COMPLETED),
            step,
        ], message="ok", status=ExecutionStatus.RUNNING)

        graph = build_main_graph(
            planner_llm=planner_llm,
            react_graph=CapturingReactGraph(),
            json_parser=mock_json_parser,
            summary_llm=planner_llm,
            uow_factory=MagicMock(),
            session_id="sess-exec-2",
        )

        history_messages = [
            {"role": "system", "content": "old system prompt"},
            {"role": "user", "content": "old execution prompt"},
            {"role": "assistant", "content": '{"success": true, "result": "collected data", "attachments": []}'},
        ]

        await graph.ainvoke({
            "message": "continue",
            "language": "zh",
            "attachments": [],
            "plan": plan,
            "current_step": step,
            "messages": history_messages,
            "execution_summary": "",
            "events": [],
            "flow_status": "executing",
            "session_id": "sess-exec-2",
            "should_interrupt": False,
            "original_request": "analyze data",
            "skill_context": "",
            "conversation_summaries": [],
            "is_resuming": False,
        })

        assert len(captured_react_inputs) >= 1
        msgs = captured_react_inputs[0]["messages"]
        # System prompt should be updated (not "old system prompt")
        assert msgs[0]["role"] == "system"
        assert "任务执行智能体" in msgs[0]["content"]
        # Should NOT contain "用户已完成接管" (not resuming)
        all_content = " ".join(m.get("content", "") for m in msgs)
        assert "用户已完成接管" not in all_content

    async def test_resuming_uses_takeover_message(self, mock_json_parser):
        """When is_resuming=True with saved messages, executor appends takeover resume message."""
        from app.domain.services.graphs.main_graph import build_main_graph
        from app.domain.models.plan import Plan, Step, ExecutionStatus

        captured_react_inputs = []

        class CapturingReactGraph:
            async def astream(self, input_state, config=None):
                captured_react_inputs.append(input_state)
                yield {"llm_node": {
                    "events": [],
                    "messages": input_state["messages"] + [
                        {"role": "assistant", "content": '{"success": true, "result": "done", "attachments": []}'}
                    ],
                    "should_interrupt": False,
                }}

        planner_llm = AsyncMock()
        planner_llm.invoke = AsyncMock()

        step = Step(description="Login to Notion")
        plan = Plan(title="T", goal="G", language="zh", steps=[step], message="ok", status=ExecutionStatus.RUNNING)

        graph = build_main_graph(
            planner_llm=planner_llm,
            react_graph=CapturingReactGraph(),
            json_parser=mock_json_parser,
            summary_llm=planner_llm,
            uow_factory=MagicMock(),
            session_id="sess-resume",
        )

        saved = [
            {"role": "system", "content": "some system prompt"},
            {"role": "user", "content": "do login"},
        ]

        await graph.ainvoke({
            "message": "我已经登录了",
            "language": "zh",
            "attachments": [],
            "plan": plan,
            "current_step": step,
            "messages": saved,
            "execution_summary": "",
            "events": [],
            "flow_status": "executing",
            "session_id": "sess-resume",
            "should_interrupt": False,
            "original_request": "login",
            "skill_context": "",
            "conversation_summaries": [],
            "is_resuming": True,
        })

        assert len(captured_react_inputs) >= 1
        msgs = captured_react_inputs[0]["messages"]
        all_content = " ".join(m.get("content", "") for m in msgs)
        assert "用户已完成接管" in all_content
```

**Step 2: Run tests to verify they fail**

Run: `cd api && python -m pytest tests/domain/services/graphs/test_main_graph.py::TestExecutorMessageBranching -v`

Expected: FAIL — second test fails because "用户已完成接管" appears when it shouldn't

**Step 3: Implement the three-branch logic**

In `api/app/domain/services/graphs/main_graph.py`, replace lines 155-171 in `executor_node`:

```python
        # 三分支 messages 构建逻辑
        is_resuming = state.get("is_resuming", False)
        saved_messages = state.get("messages") or []

        if is_resuming and saved_messages:
            # 中断恢复：保留消息 + 追加恢复提示
            initial_messages = saved_messages + [
                {"role": "user", "content": f"用户已完成接管并交还控制。请继续执行当前步骤：{step.description}\n用户消息：{state['message']}"},
            ]
        elif saved_messages:
            # 非首步/有历史：更新 system prompt 为最新版本，追加新 execution prompt
            saved_messages[0]["content"] = system_content
            initial_messages = saved_messages + [
                {"role": "user", "content": EXECUTION_PROMPT.format(
                    message=state["message"],
                    attachments=", ".join(attachments) if attachments else "无",
                    language=language,
                    step=step.description,
                )},
            ]
        else:
            # 首步/无历史：干净的 system + execution prompt
            initial_messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": EXECUTION_PROMPT.format(
                    message=state["message"],
                    attachments=", ".join(attachments) if attachments else "无",
                    language=language,
                    step=step.description,
                )},
            ]
```

**Step 4: Run tests to verify they pass**

Run: `cd api && python -m pytest tests/domain/services/graphs/test_main_graph.py -v`

Expected: ALL PASS

**Step 5: Commit**

```bash
git add api/app/domain/services/graphs/main_graph.py api/tests/domain/services/graphs/test_main_graph.py
git commit -m "fix: executor three-branch message logic to distinguish resume vs history vs fresh"
```

---

### Task 5: Fix step success detection

**Files:**
- Modify: `api/app/domain/services/graphs/main_graph.py` (executor_node, lines 197-199)
- Test: `api/tests/domain/services/graphs/test_main_graph.py`

**Step 1: Write the failing test**

In `api/tests/domain/services/graphs/test_main_graph.py`, add to `TestMainGraphFlow`:

```python
async def test_step_success_false_when_llm_reports_failure(self, mock_json_parser):
    """When react LLM returns success=false, the step should be marked as failed."""
    from app.domain.services.graphs.main_graph import build_main_graph

    class FailingReactGraph:
        async def astream(self, input_state, config=None):
            yield {"llm_node": {
                "events": [],
                "messages": input_state["messages"] + [
                    {"role": "assistant", "content": '{"success": false, "result": "CAPTCHA blocked", "attachments": []}'}
                ],
                "should_interrupt": False,
            }}

    planner_llm = AsyncMock()
    async def mock_invoke(**kwargs):
        return {
            "content": '{"title":"T","goal":"G","language":"zh","steps":[{"description":"search news"}],"message":"ok"}',
            "role": "assistant",
        }
    planner_llm.invoke = mock_invoke

    graph = build_main_graph(
        planner_llm=planner_llm,
        react_graph=FailingReactGraph(),
        json_parser=mock_json_parser,
        summary_llm=planner_llm,
        uow_factory=MagicMock(),
        session_id="sess-fail",
    )

    result = await graph.ainvoke({
        "message": "search AI news",
        "language": "zh",
        "attachments": [],
        "plan": None,
        "current_step": None,
        "messages": [],
        "execution_summary": "",
        "events": [],
        "flow_status": "idle",
        "session_id": "sess-fail",
        "should_interrupt": False,
        "original_request": "",
        "skill_context": "",
        "conversation_summaries": [],
        "is_resuming": False,
    })

    plan = result.get("plan")
    assert plan is not None
    completed_steps = [s for s in plan.steps if s.status == ExecutionStatus.COMPLETED]
    assert len(completed_steps) >= 1
    assert completed_steps[0].success is False
```

**Step 2: Run test to verify it fails**

Run: `cd api && python -m pytest tests/domain/services/graphs/test_main_graph.py::TestMainGraphFlow::test_step_success_false_when_llm_reports_failure -v`

Expected: FAIL — `step.success` is True

**Step 3: Implement the fix**

In `api/app/domain/services/graphs/main_graph.py`, add `import json` at top if not present, then replace lines 197-217:

```python
        # Extract execution summary and detect step success from LLM response JSON
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
                    pass  # Non-JSON response treated as success
                break

        step.status = ExecutionStatus.COMPLETED
        step.success = step_success
        await _emit(StepEvent(step=step, status=StepEventStatus.COMPLETED))
```

**Step 4: Run tests to verify they pass**

Run: `cd api && python -m pytest tests/domain/services/graphs/test_main_graph.py -v`

Expected: ALL PASS

**Step 5: Commit**

```bash
git add api/app/domain/services/graphs/main_graph.py api/tests/domain/services/graphs/test_main_graph.py
git commit -m "fix: detect step success/failure from LLM response JSON instead of hardcoding True"
```

---

### Task 6: Summarizer generates final summary via LLM

**Files:**
- Modify: `api/app/domain/services/graphs/main_graph.py` (summarizer_node)
- Test: `api/tests/domain/services/graphs/test_main_graph.py`

**Step 1: Write the failing test**

In `api/tests/domain/services/graphs/test_main_graph.py`, add to `TestMainGraphFlow`:

```python
async def test_summarizer_emits_message_event(self, mock_json_parser):
    """Summarizer should call LLM and emit a MessageEvent with the summary."""
    from app.domain.services.graphs.main_graph import build_main_graph
    from app.domain.models.plan import Plan, Step, ExecutionStatus

    summary_llm = AsyncMock()
    async def mock_summary_invoke(**kwargs):
        return {
            "content": '{"message": "任务已完成，这是你的总结报告。", "attachments": ["/home/ubuntu/report.md"]}',
            "role": "assistant",
        }
    summary_llm.invoke = mock_summary_invoke

    planner_llm = AsyncMock()
    planner_llm.invoke = AsyncMock()

    plan = Plan(
        title="T", goal="G", language="zh",
        steps=[Step(description="done step", status=ExecutionStatus.COMPLETED)],
        message="ok", status=ExecutionStatus.RUNNING,
    )

    graph = build_main_graph(
        planner_llm=planner_llm,
        react_graph=_make_mock_react_graph(),
        json_parser=mock_json_parser,
        summary_llm=summary_llm,
        uow_factory=MagicMock(),
        session_id="sess-sum",
    )

    result = await graph.ainvoke({
        "message": "summarize",
        "language": "zh",
        "attachments": [],
        "plan": plan,
        "current_step": None,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": '{"success": true, "result": "done", "attachments": []}'},
        ],
        "execution_summary": "",
        "events": [],
        "flow_status": "summarizing",
        "session_id": "sess-sum",
        "should_interrupt": False,
        "original_request": "G",
        "skill_context": "",
        "conversation_summaries": [],
        "is_resuming": False,
    })

    events = result.get("events", [])
    msg_events = [e for e in events if isinstance(e, MessageEvent)]
    assert len(msg_events) >= 1
    assert "总结报告" in msg_events[0].message or "任务已完成" in msg_events[0].message
```

**Step 2: Run test to verify it fails**

Run: `cd api && python -m pytest tests/domain/services/graphs/test_main_graph.py::TestMainGraphFlow::test_summarizer_emits_message_event -v`

Expected: FAIL — no MessageEvent from summarizer

**Step 3: Implement the fix**

In `api/app/domain/services/graphs/main_graph.py`, modify `summarizer_node`:

```python
    async def summarizer_node(state: MainGraphState) -> dict:
        """Generate final summary and emit completion events."""
        from app.domain.services.prompts.react import SUMMARIZE_PROMPT

        plan = state["plan"]
        events = []

        if plan:
            plan.status = ExecutionStatus.COMPLETED
            events.append(PlanEvent(plan=plan, status=PlanEventStatus.COMPLETED))

        # Call LLM to generate a user-facing summary
        react_messages = state.get("messages", [])
        if react_messages:
            try:
                response = await summary_llm.invoke(
                    messages=react_messages + [
                        {"role": "user", "content": SUMMARIZE_PROMPT},
                    ],
                )
                summary_content = response.get("content", "")
                if summary_content:
                    # Try to parse JSON response for structured output
                    try:
                        parsed = json.loads(summary_content)
                        if isinstance(parsed, dict) and parsed.get("message"):
                            events.append(MessageEvent(
                                role="assistant",
                                message=parsed["message"],
                            ))
                        else:
                            events.append(MessageEvent(role="assistant", message=summary_content))
                    except (json.JSONDecodeError, TypeError):
                        events.append(MessageEvent(role="assistant", message=summary_content))
            except Exception as exc:
                logger.warning(f"Summarizer LLM call failed: {exc}")

        events.append(DoneEvent())

        return {
            "flow_status": "completed",
            "events": events,
        }
```

Note: `json` import should already be at the top of the file (added in Task 5). Also add `import json` to the top imports if not already present.

**Step 4: Run tests to verify they pass**

Run: `cd api && python -m pytest tests/domain/services/graphs/test_main_graph.py -v`

Expected: ALL PASS

**Step 5: Commit**

```bash
git add api/app/domain/services/graphs/main_graph.py api/tests/domain/services/graphs/test_main_graph.py
git commit -m "fix: summarizer calls LLM to generate final summary instead of empty DoneEvent"
```

---

### Task 7: Run full integration test suite and fix regressions

**Files:**
- Test: `api/tests/domain/services/graphs/test_integration.py`
- Possibly modify: any file with regressions

**Step 1: Run all graph tests**

Run: `cd api && python -m pytest tests/domain/services/graphs/ -v`

Expected: ALL PASS. If any fail, fix the regressions.

**Step 2: Run full backend test suite**

Run: `cd api && python -m pytest tests/ -v --timeout=30`

Expected: ALL PASS (or pre-existing failures only)

**Step 3: Commit any regression fixes**

```bash
git add -u
git commit -m "fix: address test regressions from context and completion fixes"
```

(Skip this commit if no regressions.)
