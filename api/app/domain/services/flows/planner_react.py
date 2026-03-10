"""Planner+ReAct Flow — now delegates to LangGraph main_graph + react_graph.

This module preserves the same public interface (constructor, invoke, done)
so that AgentTaskRunner requires minimal changes.
"""

import logging
from typing import AsyncGenerator, Callable, Optional

from app.domain.external.browser import Browser
from app.domain.external.json_parser import JSONParser
from app.domain.external.llm import LLM
from app.domain.external.sandbox import Sandbox
from app.domain.external.search import SearchEngine
from app.domain.models.app_config import AgentConfig
from app.domain.models.context_overflow_config import ContextOverflowConfig
from app.domain.models.conversation_summary import ConversationSummary
from app.domain.models.event import BaseEvent, DoneEvent, WaitEvent
from app.domain.models.memory import Memory
from app.domain.models.message import Message
from app.domain.models.plan import ExecutionStatus, Plan
from app.domain.repositories.uow import IUnitOfWork
from app.domain.services.graphs.event_bridge import GraphEventBridge
from app.domain.services.graphs.main_graph import build_main_graph
from app.domain.services.graphs.react_graph import build_react_graph
from app.domain.services.tools.a2a import A2ATool
from app.domain.services.tools.base import BaseTool
from app.domain.services.tools.langchain_mcp import create_mcp_langchain_tools
from app.domain.services.tools.langchain_skill_tools import create_skill_langchain_tools
from app.domain.services.tools.langchain_tools import create_native_tools
from app.domain.services.tools.mcp import MCPTool
from app.domain.services.tools.skill import SkillTool
from app.infrastructure.external.llm.langchain_adapter import LLMAdapter

from .base import BaseFlow, FlowStatus
from .skill_creation_graph import SkillCreationGraph
from .skill_graph_canary import is_skill_graph_enabled

logger = logging.getLogger(__name__)


class PlannerReActFlow(BaseFlow):
    """Planner+ReAct orchestration flow backed by LangGraph."""

    def __init__(
        self,
        uow_factory: Callable[[], IUnitOfWork],
        llm: LLM,
        agent_config: AgentConfig,
        session_id: str,
        json_parser: JSONParser,
        browser: Browser,
        sandbox: Sandbox,
        search_engine: SearchEngine,
        mcp_tool: MCPTool,
        a2a_tool: A2ATool,
        skill_tool: SkillTool,
        create_skill_tool: BaseTool | None = None,
        brainstorm_skill_tool: BaseTool | None = None,
        overflow_config: ContextOverflowConfig | None = None,
        summary_llm: LLM | None = None,
        user_id: str = "",
        skill_graph_canary_percent: int = 0,
    ) -> None:
        self._uow_factory = uow_factory
        self._session_id = session_id
        self._json_parser = json_parser
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

        # Build LangChain tools
        lc_tools = create_native_tools(
            sandbox=sandbox, browser=browser, search_engine=search_engine,
        )
        lc_tools.extend(create_mcp_langchain_tools(mcp_tool))
        lc_tools.extend(create_skill_langchain_tools(
            brainstorm_skill_tool=brainstorm_skill_tool,
            create_skill_tool=create_skill_tool,
        ))

        # 中断恢复上下文：保存中断时的步骤、消息历史、原始请求
        self._saved_current_step = None
        self._saved_messages: list = []
        self._saved_original_request: str = ""

        # Build LLM adapter
        self._llm_adapter = LLMAdapter(llm=llm)

        # Build graphs
        self._react_graph = build_react_graph(
            llm=self._llm_adapter, tools=lc_tools, agent_config=agent_config,
        )
        self._main_graph = build_main_graph(
            planner_llm=llm,
            react_graph=self._react_graph,
            json_parser=json_parser,
            summary_llm=self._summary_llm,
            uow_factory=uow_factory,
            session_id=session_id,
            agent_config=agent_config,
        )

    def set_skill_context(self, skill_context: str) -> None:
        """Set activated skill context for this round."""
        self._skill_context = skill_context

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
            for s in plan.steps
        )
        prompt = GENERATE_SUMMARY_PROMPT.format(
            round_number=len(existing) + 1,
            plan_goal=plan.goal,
            steps_summary=steps_summary,
        )
        response = await self._summary_llm.invoke(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        parsed = await self._json_parser.invoke(response.get("content", ""))
        if not isinstance(parsed, dict):
            parsed = {}
        return ConversationSummary(
            round_number=len(existing) + 1,
            user_intent=parsed.get("user_intent", plan.goal),
            plan_summary=parsed.get("plan_summary", ""),
            execution_results=parsed.get("execution_results", []),
            decisions=parsed.get("decisions", []),
            unresolved=parsed.get("unresolved", []),
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
        """Try to drive skill creation subgraph. Returns async generator or None."""
        if not self._is_skill_graph_active():
            return None
        if not self._brainstorm_skill_tool or not self._create_skill_tool:
            return None

        action = message.skill_confirmation_action

        async with self._uow_factory() as uow:
            graph_state = await uow.session.get_skill_graph_state(self._session_id)

        if graph_state is None and action is None:
            return None
        if graph_state is not None and graph_state.is_terminal:
            return None

        async def _drive() -> AsyncGenerator[BaseEvent, None]:
            graph = SkillCreationGraph(
                brainstorm_tool=self._brainstorm_skill_tool,
                create_skill_tool=self._create_skill_tool,
            )
            async for event in graph.run(
                state=graph_state,
                action=action,
                original_request=getattr(graph_state, "original_request", ""),
            ):
                yield event

            new_state = graph.state
            if new_state is not None:
                async with self._uow_factory() as uow:
                    if new_state.is_terminal:
                        await uow.session.clear_skill_graph_state(self._session_id)
                    else:
                        await uow.session.save_skill_graph_state(
                            self._session_id, new_state,
                        )

        return _drive()

    async def invoke(self, message: Message) -> AsyncGenerator[BaseEvent, None]:
        """Run the flow — delegates to LangGraph main_graph."""
        # Try skill creation subgraph first
        subgraph_gen = await self._try_drive_skill_graph(message)
        if subgraph_gen is not None:
            async for event in subgraph_gen:
                yield event
            return

        # Build input state for main_graph
        # 如果上次因中断（接管）暂停，恢复保存的上下文
        is_resuming = self.status == FlowStatus.EXECUTING and self.plan is not None

        # === Before Graph: 加载记忆和摘要 ===
        async with self._uow_factory() as uow:
            memory = await uow.session.get_memory(self._session_id, "react")
            summaries = await uow.session.get_summary(self._session_id)

        # 上下文锚点：非首轮 + 配置开启时注入
        if self._memory_config.context_anchor_enabled and not memory.empty:
            anchor = self._build_context_anchor(message)
            memory.add_message({"role": "user", "content": anchor})

        # 摘要文本
        recent_summaries = summaries[-self._memory_config.summary_max_rounds:]
        summary_texts = [s.to_prompt_text() for s in recent_summaries]

        input_state = {
            "message": message.message,
            "language": getattr(message, "language", "zh"),
            "attachments": getattr(message, "attachments", []),
            "plan": self.plan,
            "current_step": self._saved_current_step if is_resuming else None,
            "messages": self._saved_messages if is_resuming else memory.get_messages(),
            "execution_summary": "",
            "events": [],
            "flow_status": "executing" if is_resuming else (
                self.status.value if hasattr(self.status, "value") else "idle"
            ),
            "session_id": self._session_id,
            "should_interrupt": False,
            "is_resuming": is_resuming,
            "original_request": self._saved_original_request if is_resuming else "",
            "skill_context": self._skill_context,
            "conversation_summaries": summary_texts,
        }

        bridge = GraphEventBridge()
        async for event in bridge.run(self._main_graph, input_state):
            yield event

        # Update internal state from graph result
        final = bridge.final_state
        self.plan = final.get("plan")
        flow_status = final.get("flow_status", "idle")

        if final.get("should_interrupt"):
            # 中断（接管）：保存上下文以便恢复
            self.status = FlowStatus.EXECUTING
            self._saved_current_step = final.get("current_step")
            self._saved_messages = final.get("messages", [])
            self._saved_original_request = final.get("original_request", "")
        else:
            # === After Graph: 记忆压缩、保存、摘要生成 ===
            react_messages = final.get("messages", [])
            memory = Memory(messages=list(react_messages))
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
            self._saved_current_step = None
            self._saved_messages = []
            self._saved_original_request = ""

    @property
    def done(self) -> bool:
        return self.status == FlowStatus.IDLE
