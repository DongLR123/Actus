import logging

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)


class CheckpointerPool:
    """Application-scoped AsyncConnectionPool for LangGraph checkpointers.

    Manages pool lifecycle (open/close) and runs checkpointer DDL migrations
    once at startup. Exposes the raw pool for injection into PlannerReActFlow.
    """

    def __init__(
        self,
        db_url: str,
        min_size: int = 2,
        max_size: int = 10,
        timeout: float = 30.0,
    ) -> None:
        conninfo = db_url.replace("+asyncpg", "")
        self._pool = AsyncConnectionPool(
            conninfo=conninfo,
            min_size=min_size,
            max_size=max_size,
            open=False,
            timeout=timeout,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
        )

    async def open(self) -> None:
        """Open the pool and run checkpointer DDL migrations."""
        await self._pool.open(wait=True)
        try:
            checkpointer = AsyncPostgresSaver(self._pool)
            await checkpointer.setup()
            logger.info("Checkpointer pool opened and migrations completed")
        except Exception:
            await self._pool.close()
            raise

    async def close(self) -> None:
        """Close the pool, rejecting new checkouts and waiting for returns."""
        await self._pool.close()
        logger.info("Checkpointer pool closed")

    @property
    def pool(self) -> AsyncConnectionPool:
        return self._pool
