"""Bridge between LangGraph execution and Actus Event stream.

The bridge runs a compiled LangGraph, extracts accumulated events from the
final state, and yields them as an AsyncGenerator — matching the interface
that AgentTaskRunner expects.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator

from app.domain.models.event import BaseEvent


class GraphEventBridge:
    """Runs a LangGraph and streams the accumulated events."""

    def __init__(self) -> None:
        self._final_state: dict[str, Any] = {}

    @property
    def final_state(self) -> dict[str, Any]:
        """The full graph output state after execution."""
        return self._final_state

    async def run(
        self,
        graph: Any,
        input_state: dict[str, Any],
    ) -> AsyncGenerator[BaseEvent, None]:
        """Invoke the graph and yield events from the result.

        Parameters
        ----------
        graph : Compiled LangGraph (has `ainvoke` method).
        input_state : Initial state dict for the graph.
        """
        self._final_state = await graph.ainvoke(input_state)

        for event in self._final_state.get("events", []):
            yield event
