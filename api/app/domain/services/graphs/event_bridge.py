"""Bridge between LangGraph execution and Actus Event stream.

Uses an asyncio.Queue so that events from both the main graph nodes and
the nested react_graph are yielded to the frontend in real-time.

Nodes that receive an ``event_queue`` via LangGraph config can push events
directly; other nodes' events are picked up from astream output.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator

from app.domain.models.event import BaseEvent

logger = logging.getLogger(__name__)


class GraphEventBridge:
    """Runs a LangGraph and streams events in real-time via an async queue."""

    def __init__(self) -> None:
        self._final_state: dict[str, Any] = {}
        self._was_interrupted: bool = False

    @property
    def final_state(self) -> dict[str, Any]:
        """The full graph output state after execution."""
        return self._final_state

    @property
    def was_interrupted(self) -> bool:
        """Whether the graph was interrupted (has pending interrupt_node)."""
        return self._was_interrupted

    async def run(
        self,
        graph: Any,
        input_state: dict[str, Any] | Any,
        config: dict[str, Any] | None = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Stream the graph, yielding events as each node produces them.

        Events reach the caller via two paths:

        1. **Queue path** — nodes that receive ``event_queue`` in config push
           events directly (used by ``executor_node`` for react sub-graph).
        2. **State path** — nodes that only return ``{"events": [...]}`` have
           their events picked up here from the ``astream`` output.

        Parameters
        ----------
        graph : Compiled LangGraph StateGraph.
        input_state : Input state dict, or a Command for resuming.
        config : Optional config dict with ``configurable`` keys (e.g. thread_id).
        """
        queue: asyncio.Queue[BaseEvent | None] = asyncio.Queue()
        # 用 input_state 初始化 _final_state，确保节点未返回的字段保留输入值
        # （例如 executor_node 跳过时不会丢失 messages）
        # 仅当 input_state 为 dict 时初始化（Command 不可展开为 dict）
        if isinstance(input_state, dict):
            self._final_state = dict(input_state)
        else:
            self._final_state = {}

        # Merge event_queue into config
        merged_config: dict[str, Any] = {"configurable": {"event_queue": queue}}
        if config:
            for key, value in config.items():
                if key == "configurable":
                    merged_config["configurable"].update(value)
                else:
                    merged_config[key] = value

        async def _drive_graph() -> None:
            """Run the graph and forward state-path events to the queue."""
            try:
                async for chunk in graph.astream(
                    input_state,
                    config=merged_config,
                ):
                    for _node_name, node_output in chunk.items():
                        if not isinstance(node_output, dict):
                            continue
                        self._final_state.update(node_output)
                        # Emit events that were NOT already pushed via queue
                        # (nodes using queue return events=[])
                        for evt in node_output.get("events") or []:
                            if isinstance(evt, BaseEvent):
                                await queue.put(evt)
            except Exception:
                logger.exception("GraphEventBridge: graph execution error")
                raise
            finally:
                # Detect if graph was interrupted via checkpointer state
                try:
                    graph_state = await graph.aget_state(merged_config)
                    if graph_state and graph_state.next:
                        self._was_interrupted = True
                except Exception:
                    # No checkpointer or aget_state unavailable (e.g. mock graph) —
                    # fall back to checking should_interrupt in final state
                    self._was_interrupted = self._final_state.get(
                        "should_interrupt", False
                    )
                await queue.put(None)  # sentinel

        task = asyncio.create_task(_drive_graph())

        # Track whether the consumer exited normally (sentinel received)
        # vs via GeneratorExit (caller closed the generator, e.g. after WaitEvent).
        # In the cleanup path, suppress _drive_graph exceptions to prevent
        # WAITING → COMPLETED overwrite in agent_task_runner's except handler.
        _normal_exit = False
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
            _normal_exit = True
        finally:
            if _normal_exit:
                await task
            else:
                try:
                    await task
                except Exception:
                    # Suppress _drive_graph exceptions during cleanup.
                    # This is critical: when agent_task_runner processes WaitEvent
                    # and returns, the generator cleanup chain runs. If _drive_graph
                    # raises (e.g. interrupt() error), the exception propagates to
                    # agent_task_runner's `except Exception` handler, which overwrites
                    # WAITING status to COMPLETED. Errors are already logged by
                    # _drive_graph's own exception handler.
                    logger.warning(
                        "GraphEventBridge: suppressed _drive_graph error during "
                        "generator cleanup (error already logged above)"
                    )
