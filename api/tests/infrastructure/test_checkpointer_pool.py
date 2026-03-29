import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.infrastructure.checkpointer_pool import CheckpointerPool


def test_init_creates_closed_pool():
    """CheckpointerPool should create an AsyncConnectionPool in closed state."""
    cp = CheckpointerPool(
        db_url="postgresql+asyncpg://user:pass@localhost/db",
        min_size=1,
        max_size=5,
        timeout=10.0,
    )
    assert cp.pool is not None
    assert cp.pool.closed


def test_init_strips_asyncpg_driver():
    """db_url with +asyncpg driver should be converted to plain postgresql://."""
    cp = CheckpointerPool(
        db_url="postgresql+asyncpg://user:pass@localhost/db",
    )
    assert "+asyncpg" not in cp.pool.conninfo


@pytest.mark.anyio
async def test_open_calls_pool_open_and_setup():
    """open() should open the pool with wait=True, then run checkpointer.setup()."""
    cp = CheckpointerPool(db_url="postgresql://user:pass@localhost/db")

    mock_pool_open = AsyncMock()
    mock_setup = AsyncMock()

    with patch.object(cp._pool, "open", mock_pool_open), \
         patch("app.infrastructure.checkpointer_pool.AsyncPostgresSaver") as MockSaver:
        MockSaver.return_value.setup = mock_setup
        await cp.open()

    mock_pool_open.assert_awaited_once_with(wait=True)
    mock_setup.assert_awaited_once()


@pytest.mark.anyio
async def test_open_closes_pool_on_setup_failure():
    """If setup() raises, open() should close the pool and re-raise."""
    cp = CheckpointerPool(db_url="postgresql://user:pass@localhost/db")

    mock_pool_close = AsyncMock()

    with patch.object(cp._pool, "open", AsyncMock()), \
         patch.object(cp._pool, "close", mock_pool_close), \
         patch("app.infrastructure.checkpointer_pool.AsyncPostgresSaver") as MockSaver:
        MockSaver.return_value.setup = AsyncMock(side_effect=RuntimeError("migration failed"))

        with pytest.raises(RuntimeError, match="migration failed"):
            await cp.open()

    mock_pool_close.assert_awaited_once()


@pytest.mark.anyio
async def test_close_calls_pool_close():
    """close() should delegate to the underlying pool."""
    cp = CheckpointerPool(db_url="postgresql://user:pass@localhost/db")

    mock_pool_close = AsyncMock()
    with patch.object(cp._pool, "close", mock_pool_close):
        await cp.close()

    mock_pool_close.assert_awaited_once()
