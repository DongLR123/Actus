"""Planner+ReAct Flow — now delegates to LangGraph main_graph + react_graph.

This module preserves the same public interface (constructor, invoke, done)
so that AgentTaskRunner requires minimal changes.
"""

import logging
from typing import Any, AsyncGenerator, Callable, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.domain.external.browser import Browser
from app.domain.external.sandbox import Sandbox
from app.domain.external.search import SearchEngine
from app.domain.models.app_config import AgentConfig
from app.domain.models.context_overflow_config import ContextOverflowConfig
from app.domain.models.conversation_summary import ConversationSummary
from app.domain.models.event import (
    BaseEvent,
    DoneEvent,
    MessageEvent,
    PlanEvent,
    PlanEventStatus,
    TitleEvent,
    WaitEvent,
)
from app.domain.models.llm_responses import ConversationSummaryResponse, PlanResponse
from app.domain.models.memory import Memory
from app.domain.models.message import Message
from app.domain.models.plan import ExecutionStatus, Plan, Step
from app.domain.repositories.uow import IUnitOfWork
from langgraph.types import Command

from app.domain.services.graphs.event_bridge import GraphEventBridge
from app.domain.services.graphs.main_graph import build_main_graph
from app.domain.services.graphs.message_utils import dicts_to_messages, messages_to_dicts
from app.domain.services.graphs.react_graph import build_react_graph
from app.domain.services.tools.a2a import A2ATool
from app.domain.services.tools.base import BaseTool
from app.domain.services.tools.langchain_mcp import create_mcp_langchain_tools
from app.domain.services.tools.langchain_skill_tools import create_skill_langchain_tools
from app.domain.services.tools.langchain_tools import create_native_tools
from app.domain.services.tools.mcp import MCPTool
from app.domain.services.tools.skill import SkillTool

from .base import BaseFlow, FlowStatus
from .skill_creation_graph import SkillCreationGraph
from .skill_graph_canary import is_skill_graph_enabled

logger = logging.getLogger(__name__)


class PlannerReActFlow(BaseFlow):
    """Planner+ReAct orchestration flow backed by LangGraph."""

    def __init__(
        self,
        uow_factory: Callable[[], IUnitOfWork],
        llm: BaseChatModel,
        agent_config: AgentConfig,
        session_id: str,
        browser: Browser,
        sandbox: Sandbox,
        search_engine: SearchEngine,
        mcp_tool: MCPTool,
        a2a_tool: A2ATool,
        skill_tool: SkillTool,
        create_skill_tool: BaseTool | None = None,
        brainstorm_skill_tool: BaseTool | None = None,
        overflow_config: ContextOverflowConfig | None = None,
        summary_llm: BaseChatModel | None = None,
        user_id: str = "",
        skill_graph_canary_percent: int = 0,
        db_url: str = "",
        checkpointer: Any = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._session_id = session_id
        self._summary_llm = summary_llm or llm
        self.status = FlowStatus.IDLE
        self.plan: Optional[Plan] = None
        self._memory_config = agent_config.memory
        self._overflow_config = overflow_config
        self._skill_context = ""

        # Skill creation subgraph
        self._user_id = user_id
        self._skill_graph_canary_percent = skill_graph_canary_percent
        self._brainstorm_skill_tool = brainstorm_skill_tool
        self._create_skill_tool = create_skill_tool

        # 延迟绑定：保存依赖引用，在 invoke() 时构建工具和图
        # MCP/A2A 在 AgentTaskRunner.run() 中异步初始化，构造时尚未就绪
        self._llm = llm
        self._agent_config = agent_config
        self._sandbox = sandbox
        self._browser = browser
        self._search_engine = search_engine
        self._mcp_tool = mcp_tool
        self._a2a_tool = a2a_tool
        self._graphs_built = False
        self._react_graph = None
        self._main_graph = None

        # LangGraph checkpointer — 跨 graph 重建复用，支持 interrupt/resume
        self._checkpointer = checkpointer  # None = lazy-init AsyncPostgresSaver
        self._db_url = db_url

    async def _get_checkpointer(self):
        """Lazy-initialize checkpointer. Returns injected checkpointer (test) or AsyncPostgresSaver (prod).

        Note: The lazy import of AsyncPostgresSaver in the domain layer is a pragmatic
        DDD compromise — the checkpointer is injected in tests, and only falls back to
        infrastructure-level initialization when no checkpointer is provided.
        """
        if self._checkpointer is not None:
            return self._checkpointer
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg import AsyncConnection

        conn_string = self._db_url.replace("+asyncpg", "")
        conn = await AsyncConnection.connect(
            conn_string, autocommit=True, prepare_threshold=0,
        )
        self._checkpointer = AsyncPostgresSaver(conn=conn)
        await self._checkpointer.setup()
        return self._checkpointer

    def set_skill_context(self, skill_context: str) -> None:
        """Set activated skill context for this round."""
        self._skill_context = skill_context

    async def _ensure_graphs(self) -> None:
        """延迟构建工具列表和 LangGraph 图。

        在 invoke() 首次调用时执行，此时 MCP/A2A 已完成异步初始化，
        能正确获取到所有可用工具。后续调用会重新构建以反映工具变化。
        """
        checkpointer = await self._get_checkpointer()
        from app.domain.services.tools.langchain_a2a import create_a2a_langchain_tools

        lc_tools = create_native_tools(
            sandbox=self._sandbox, browser=self._browser,
            search_engine=self._search_engine,
        )
        lc_tools.extend(create_mcp_langchain_tools(self._mcp_tool))
        lc_tools.extend(create_a2a_langchain_tools(self._a2a_tool))
        lc_tools.extend(create_skill_langchain_tools(
            brainstorm_skill_tool=self._brainstorm_skill_tool,
            create_skill_tool=self._create_skill_tool,
        ))

        tool_names = [t.name for t in lc_tools]
        logger.info("延迟绑定工具列表 (%d tools): %s", len(lc_tools), tool_names)

        self._react_graph = build_react_graph(
            llm=self._llm, tools=lc_tools, agent_config=self._agent_config,
        )
        self._main_graph = build_main_graph(
            planner_llm=self._llm,
            react_graph=self._react_graph,
            summary_llm=self._summary_llm,
            uow_factory=self._uow_factory,
            session_id=self._session_id,
            agent_config=self._agent_config,
            checkpointer=checkpointer,
        )
        self._graphs_built = True

    def _build_previous_plan_context(self) -> str:
        """构建前一轮计划的上下文摘要，包含步骤结果和生成的文件路径。

        当用户发送新消息时（如「输出为ppt」），Planner 需要知道前一轮做了什么、
        生成了哪些文件，才能正确理解用户意图并制定关联计划。
        """
        if not self.plan or not self.plan.steps:
            return ""
        lines = [f"### 前一轮任务回顾"]
        lines.append(f"- 目标：{self.plan.goal}")
        for s in self.plan.steps:
            status = "已完成" if s.done else "未完成"
            line = f"- 步骤「{s.description}」: {status}"
            if s.result:
                # result 通常是 JSON 字符串，包含 success/attachments/result 字段
                # 截取前 300 字符以控制 token 消耗
                line += f"\n  执行结果: {s.result[:300]}"
            lines.append(line)
        return "\n".join(lines)

    def _build_context_anchor(self, message: Message) -> str:
        """构建上下文锚点，注入到 Memory 中帮助 LLM 保持多轮连贯性。"""
        parts = ["[上下文回顾]"]
        if self.plan:
            parts.append(f"- 原始需求：{self.plan.goal}")
            completed = [s.description for s in self.plan.steps
                         if s.status == ExecutionStatus.COMPLETED]
            pending = [s.description for s in self.plan.steps
                       if s.status != ExecutionStatus.COMPLETED]
            if completed:
                parts.append(f"- 已完成：{'；'.join(completed)}")
            if pending:
                parts.append(f"- 待完成：{'；'.join(pending)}")
        parts.append(f"- 当前消息：{message.message}")
        return "\n".join(parts)

    async def _generate_summary(
        self, existing: list[ConversationSummary], plan: Plan,
    ) -> ConversationSummary:
        """调用 LLM 生成结构化对话摘要。"""
        from app.domain.services.prompts.summary import GENERATE_SUMMARY_PROMPT
        steps_summary = "\n".join(
            f"- {s.description}: {'完成' if s.status == ExecutionStatus.COMPLETED else '未完成'}"
            + (f"\n  结果: {s.result[:200]}" if s.result else "")
            for s in plan.steps
        )
        prompt = GENERATE_SUMMARY_PROMPT.format(
            round_number=len(existing) + 1,
            plan_goal=plan.goal,
            steps_summary=steps_summary,
        )
        messages = [HumanMessage(content=prompt)]
        structured = self._summary_llm.with_structured_output(ConversationSummaryResponse)
        parsed: ConversationSummaryResponse | None = await structured.ainvoke(messages)
        if parsed is None:
            parsed = ConversationSummaryResponse()
        return ConversationSummary(
            round_number=len(existing) + 1,
            user_intent=parsed.user_intent or plan.goal,
            plan_summary=parsed.plan_summary,
            execution_results=parsed.execution_results,
            decisions=parsed.decisions,
            unresolved=parsed.unresolved,
        )

    async def _check_overflow(self, memory: Memory) -> None:
        """检测上下文溢出，超过硬阈值时做激进压缩。"""
        if not self._overflow_config or not self._overflow_config.context_overflow_guard_enabled:
            return
        from app.domain.services.context.model_context_window import resolve_context_window
        # 使用字符估算 token（粗略：1 token ≈ 3-4 字符中英混合）
        total_chars = sum(len(str(m.get("content", ""))) for m in memory.messages)
        estimated_tokens = int(total_chars / 3 * self._overflow_config.token_safety_factor)
        window = resolve_context_window("", self._overflow_config)
        hard_limit = int(window * self._overflow_config.hard_trigger_ratio)
        if estimated_tokens > hard_limit:
            logger.warning(f"上下文溢出: ~{estimated_tokens} tokens > hard_limit {hard_limit}, 执行硬压缩")
            memory.compact(keep_summary=False)
            # 保留系统消息 + 最近 N 条
            if len(memory.messages) > 20:
                memory.messages = memory.messages[:1] + memory.messages[-19:]
            async with self._uow_factory() as uow:
                await uow.session.save_memory(self._session_id, "react", memory)

    def _is_skill_graph_active(self) -> bool:
        return is_skill_graph_enabled(self._user_id, self._skill_graph_canary_percent)

    async def _try_drive_skill_graph(
        self, message: Message,
    ) -> AsyncGenerator[BaseEvent, None] | None:
        """Drive skill creation subgraph for continuation messages.

        Only handles cases where there's an existing graph state (from a
        previous brainstorm/generate step). Initial request detection is
        done via planner output in invoke().
        """
        sg_active = self._is_skill_graph_active()
        has_tools = bool(self._brainstorm_skill_tool and self._create_skill_tool)
        if not sg_active:
            return None
        if not has_tools:
            return None

        action = message.skill_confirmation_action

        async with self._uow_factory() as uow:
            graph_state = await uow.session.get_skill_graph_state(self._session_id)

        # 无已有子图状态 → 不是续接消息，交由 invoke() 的 planner 路由处理
        if graph_state is None:
            return None
        if graph_state.is_terminal:
            return None

        # 防御性校验：检测持久化恢复后状态是否完整
        # 已知场景：CancelledError/断连导致状态部分写入或未写入，
        # DB 中仅剩 status="wait_generate"（默认值）而其余字段为空。
        if not graph_state.original_request:
            logger.warning(
                "Skill 子图状态不完整（original_request 为空），"
                "可能是持久化失败导致的残留状态，清除后走正常流程: "
                "session=%s status=%s",
                self._session_id, graph_state.status,
            )
            try:
                async with self._uow_factory() as uow:
                    await uow.session.clear_skill_graph_state(self._session_id)
            except Exception as exc:
                logger.warning("清除残留 Skill 子图状态失败: %s", exc)
            return None

        async def _drive() -> AsyncGenerator[BaseEvent, None]:
            graph = SkillCreationGraph(
                brainstorm_tool=self._brainstorm_skill_tool,
                create_skill_tool=self._create_skill_tool,
            )
            new_state, events = await graph.run(
                state=graph_state,
                action=action,
                original_request=graph_state.original_request,
            )
            # 关键：先持久化状态，再 yield 事件。
            ok = await self._persist_skill_graph_state(new_state)
            if not ok:
                yield MessageEvent(
                    role="assistant",
                    message="Skill 状态保存失败，请重试。",
                )
                return
            for event in events:
                yield event

        return _drive()

    async def _persist_skill_graph_state(self, state: Any) -> bool:
        """持久化 SkillCreationGraph 的状态（终态清除，非终态保存）。

        Returns
        -------
        bool : 持久化是否成功。调用方应检查返回值，
               持久化失败时不应继续展示蓝图等事件。
        """
        if state is None:
            return True
        try:
            async with self._uow_factory() as uow:
                if state.is_terminal:
                    await uow.session.clear_skill_graph_state(self._session_id)
                else:
                    await uow.session.save_skill_graph_state(
                        self._session_id, state,
                    )
            return True
        except BaseException as exc:
            logger.error(
                "Skill 子图状态持久化失败: session=%s %s",
                self._session_id, exc, exc_info=True,
            )
            return False

    async def _start_skill_subgraph(
        self, message: Message,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Start the skill creation subgraph for an initial request.

        Called after planner detects skill creation intent. Drives
        brainstorm_skill → WaitEvent, then persists state for continuation.
        """
        graph = SkillCreationGraph(
            brainstorm_tool=self._brainstorm_skill_tool,
            create_skill_tool=self._create_skill_tool,
        )
        new_state, events = await graph.run(
            state=None,
            action=None,
            original_request=message.message,
        )
        # 关键：先持久化状态，再 yield 事件。
        # 如果持久化失败，不展示蓝图（避免用户确认后找不到状态）。
        ok = await self._persist_skill_graph_state(new_state)
        if not ok:
            yield MessageEvent(
                role="assistant",
                message="Skill 蓝图生成成功但状态保存失败，请重试。",
            )
            return
        for event in events:
            yield event

    async def _run_planner_for_detection(
        self, message: Message, summary_texts: list[str],
    ) -> tuple[Plan, list[BaseEvent]]:
        """Run the planner LLM and return (Plan, plan_events).

        Used for intent detection before deciding whether to route to skill
        creation subgraph or normal main_graph flow. The plan is reused by
        main_graph (skipping planner_node) to avoid double LLM calls.
        """
        from app.domain.services.prompts.planner import (
            PLANNER_SYSTEM_PROMPT,
            CREATE_PLAN_PROMPT,
        )

        attachments = getattr(message, "attachments", [])
        prompt = CREATE_PLAN_PROMPT.format(
            message=message.message,
            attachments=", ".join(attachments) if attachments else "无",
        )

        system_content = PLANNER_SYSTEM_PROMPT

        # Inject tool summary (same logic as main_graph planner_node)
        skill_context = self._skill_context
        tool_summary_marker = "## Available Tool Summary"
        if tool_summary_marker in skill_context:
            tool_summary = skill_context[skill_context.index(tool_summary_marker):]
            system_content += f"\n\n{tool_summary}"

        if summary_texts:
            system_content += "\n\n## 历史对话摘要\n" + "\n\n".join(summary_texts)

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=prompt),
        ]
        structured = self._llm.with_structured_output(PlanResponse)
        parsed: PlanResponse | None = await structured.ainvoke(messages)

        if parsed is None:
            parsed = PlanResponse(
                title="Task",
                goal=message.message,
                language=getattr(message, "language", "zh"),
                steps=[],
                message="好的，我来帮你处理。",
            )

        steps = [
            Step(description=s.description)
            for s in parsed.steps
        ]
        if not steps:
            steps = [Step(description=message.message)]
        plan = Plan(
            title=parsed.title or "Task",
            goal=parsed.goal or message.message,
            language=parsed.language or getattr(message, "language", "zh"),
            steps=steps,
            message=parsed.message or "",
            status=ExecutionStatus.RUNNING,
        )

        events: list[BaseEvent] = [
            TitleEvent(title=plan.title),
            MessageEvent(role="assistant", message=plan.message),
            PlanEvent(plan=plan, status=PlanEventStatus.CREATED),
        ]

        return plan, events

    @staticmethod
    def _plan_uses_skill_creation(plan: Plan) -> bool:
        """Check if the planner's plan references skill creation tools.

        The planner prompt instructs it to output step descriptions containing
        'brainstorm_skill → generate_skill → install_skill' for skill creation
        requests. However, LLM 可能把关键词放在 message/goal 中而非 step
        description（甚至返回空 steps），因此需要同时检查多个字段。
        """
        parts: list[str] = [s.description for s in plan.steps]
        parts.append(plan.message or "")
        parts.append(plan.goal or "")
        text = " ".join(parts).lower()
        return "brainstorm_skill" in text or "generate_skill" in text

    async def invoke(self, message: Message) -> AsyncGenerator[BaseEvent, None]:
        """Run the flow — delegates to LangGraph main_graph."""
        # 1. Continuation: existing skill graph state → drive subgraph
        subgraph_gen = await self._try_drive_skill_graph(message)
        if subgraph_gen is not None:
            async for event in subgraph_gen:
                yield event
            return

        # 延迟绑定：每次 invoke 重新构建工具和图，确保 MCP/A2A 已初始化
        await self._ensure_graphs()

        # LangGraph config with thread_id for checkpointer
        config = {"configurable": {"thread_id": self._session_id}}

        # === Before Graph: load summaries ===
        async with self._uow_factory() as uow:
            summaries = await uow.session.get_summary(self._session_id)

        # Detect pending interrupt via checkpointer state
        is_resume = False
        try:
            graph_state = await self._main_graph.aget_state(config)
            if graph_state and graph_state.next:
                is_resume = True
        except Exception:
            pass

        if is_resume:
            # Resume path: checkpointer has saved full state, pass user response
            input_for_graph = Command(resume=message.message)
            logger.info(
                "通过 checkpointer 恢复中断: session=%s, resume=%s",
                self._session_id, message.message[:50],
            )
        else:
            # New task path
            async with self._uow_factory() as uow:
                memory = await uow.session.get_memory(self._session_id, "react")

            # Context anchor injection
            if self._memory_config.context_anchor_enabled and not memory.empty:
                anchor = self._build_context_anchor(message)
                memory.add_message({"role": "user", "content": anchor})

            # Summary texts
            recent_summaries = summaries[-self._memory_config.summary_max_rounds:]
            summary_texts = [s.to_prompt_text() for s in recent_summaries]

            # 注入前一轮计划上下文（防御 conversation_summaries 为空的情况）
            # 即使 summary 生成失败，self.plan 仍然保留了上一轮的完整计划和步骤结果
            previous_plan_context = self._build_previous_plan_context()
            if previous_plan_context:
                summary_texts.insert(0, previous_plan_context)

            # Convert Memory dict messages to LangChain BaseMessage
            raw_messages = memory.get_messages()
            lc_messages = dicts_to_messages(raw_messages) if raw_messages else []

            # 2. Planner-first routing: run planner, check for skill creation intent
            _skill_tools_available = (
                self._is_skill_graph_active()
                and self._brainstorm_skill_tool is not None
                and self._create_skill_tool is not None
            )
            if _skill_tools_available:
                plan, plan_events = await self._run_planner_for_detection(
                    message, summary_texts,
                )
                is_skill = self._plan_uses_skill_creation(plan)
                if is_skill:
                    # Emit plan events (title, message, plan) then start subgraph
                    for event in plan_events:
                        yield event
                    async for event in self._start_skill_subgraph(message):
                        yield event
                    return

                # Not skill creation: reuse pre-computed plan, skip planner_node
                # by setting flow_status=executing
                input_for_graph = {
                    "message": message.message,
                    "language": getattr(message, "language", "zh"),
                    "attachments": getattr(message, "attachments", []),
                    "plan": plan,
                    "current_step": plan.get_next_step(),
                    "messages": lc_messages,
                    "execution_summary": "",
                    "events": [],
                    "flow_status": FlowStatus.EXECUTING.value,
                    "session_id": self._session_id,
                    "should_interrupt": False,
                    "resume_value": None,
                    "original_request": plan.goal,
                    "skill_context": self._skill_context,
                    "conversation_summaries": summary_texts,
                }
                # Emit pre-computed plan events before bridge
                for event in plan_events:
                    yield event
            else:
                # No skill creation tools → normal flow with planner_node
                input_for_graph = {
                    "message": message.message,
                    "language": getattr(message, "language", "zh"),
                    "attachments": getattr(message, "attachments", []),
                    "plan": self.plan,
                    "current_step": None,
                    "messages": lc_messages,
                    "execution_summary": "",
                    "events": [],
                    "flow_status": self.status.value if hasattr(self.status, "value") else FlowStatus.IDLE.value,
                    "session_id": self._session_id,
                    "should_interrupt": False,
                    "resume_value": None,
                    "original_request": self.plan.goal if self.plan else "",
                    "skill_context": self._skill_context,
                    "conversation_summaries": summary_texts,
                }

        bridge = GraphEventBridge()
        try:
            async for event in bridge.run(self._main_graph, input_for_graph, config=config):
                yield event
        finally:
            # 使用 try/finally 确保持久化逻辑始终执行，即使消费方提前退出
            # （例如 WaitEvent 触发 agent_task_runner.run() 的 return，
            #  导致本 async generator 被 aclose()、GeneratorExit 抛入 yield 处）。
            # bridge.run() 的 finally 会 await 图任务完成，
            # 因此此处 bridge.final_state 已包含完整的图输出。
            await self._persist_after_graph(bridge.final_state, summaries)

    async def _persist_after_graph(
        self, final: dict, summaries: list[ConversationSummary],
    ) -> None:
        """Post-graph persistence: save memory and summaries.

        CRITICAL: 该方法在 invoke() 的 finally 块中调用（async generator 被 aclose
        时通过 GeneratorExit → finally 触发）。如果此方法抛出异常，异常会传播到
        agent_task_runner 的 except Exception 处理器。
        因此整个方法用 try/except 包裹，确保永不向上抛出异常。
        """
        try:
            await self._persist_after_graph_inner(final, summaries)
        except Exception as exc:
            logger.exception(
                "持久化后处理异常（已抑制，避免破坏 generator 清理链）: %s", exc
            )

    async def _persist_after_graph_inner(
        self, final: dict, summaries: list[ConversationSummary],
    ) -> None:
        """Inner implementation of post-graph persistence.

        中断路径：checkpointer 已自动保存完整图状态，无需手动序列化 messages/step。
        仅保存 Memory（用于上下文锚点）和更新 flow 状态。
        """
        self.plan = final.get("plan")

        # 将 LangChain BaseMessage 转回 dict 用于 Memory 持久化
        raw_messages = final.get("messages", [])
        try:
            dict_messages = messages_to_dicts(raw_messages) if raw_messages else []
        except Exception as exc:
            logger.warning("messages_to_dicts 转换失败，使用空消息列表: %s", exc)
            dict_messages = []

        if final.get("should_interrupt"):
            # Checkpointer has automatically saved full graph state for Command(resume=...).
            # Only persist Memory (for context anchors) and update flow status.
            self.status = FlowStatus.EXECUTING
            self.plan = final.get("plan")

            logger.info(
                "中断持久化 (checkpointer): session=%s plan=%s",
                self._session_id,
                self.plan.title if self.plan else "<none>",
            )

            memory = Memory(messages=list(dict_messages))
            memory.compact(keep_summary=self._memory_config.compact_keep_summary)
            try:
                async with self._uow_factory() as uow:
                    await uow.session.save_memory(self._session_id, "react", memory)
            except Exception as e:
                logger.warning(f"中断时保存 Memory 失败: {e}")
        else:
            # === After Graph: 记忆压缩、保存、摘要生成 ===
            memory = Memory(messages=list(dict_messages))
            memory.compact(keep_summary=self._memory_config.compact_keep_summary)

            try:
                async with self._uow_factory() as uow:
                    await uow.session.save_memory(self._session_id, "react", memory)
            except Exception as e:
                logger.warning(f"保存 Memory 失败: {e}")

            # ConversationSummary 生成（容错，不阻塞）
            plan = final.get("plan") or self.plan
            if (self._memory_config.summary_enabled
                    and plan
                    and len(plan.steps) >= self._memory_config.summary_min_steps):
                try:
                    new_summary = await self._generate_summary(summaries, plan)
                    all_summaries = (summaries + [new_summary])[
                        -self._memory_config.summary_max_rounds:
                    ]
                    async with self._uow_factory() as uow:
                        await uow.session.save_summary(self._session_id, all_summaries)
                except Exception as e:
                    logger.warning(f"生成对话摘要失败，不阻塞: {e}")

            # 上下文溢出检测
            try:
                await self._check_overflow(memory)
            except Exception as e:
                logger.warning(f"上下文溢出检测失败: {e}")

            # 正常完成：重置状态
            self.status = FlowStatus.IDLE

    @property
    def done(self) -> bool:
        return self.status == FlowStatus.IDLE
