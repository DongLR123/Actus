import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_app(cors_origins: str) -> FastAPI:
    """Build a minimal FastAPI app with the same CORS logic as main.py."""
    app = FastAPI()
    origins = [o.strip() for o in cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/status")
    async def status():
        return {"ok": True}

    return app


async def test_cors_allows_whitelisted_origin():
    app = _make_app("http://localhost:3000")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.options(
            "/api/status",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


async def test_cors_rejects_non_whitelisted_origin():
    app = _make_app("http://localhost:3000")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.options(
            "/api/status",
            headers={
                "Origin": "http://evil.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" not in response.headers


async def test_cors_multiple_origins():
    app = _make_app("http://localhost:3000,https://prod.example.com")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.options(
            "/api/status",
            headers={
                "Origin": "https://prod.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.headers.get("access-control-allow-origin") == "https://prod.example.com"


def test_cors_origins_parsing():
    """Unit test: verify comma-separated string is parsed correctly."""
    raw = " http://localhost:3000 , https://example.com , "
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    assert origins == ["http://localhost:3000", "https://example.com"]
