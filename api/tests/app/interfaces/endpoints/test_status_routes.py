import httpx
import pytest

from app.application.services.status_service import StatusService
from app.domain.models.health_status import HealthStatus
from app.domain.models.user import User, UserRole, UserStatus
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.service_dependencies import get_status_service
from app.main import app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeStatusService(StatusService):
    def __init__(self) -> None:
        super().__init__(checkers=[])

    async def check_all(self) -> list[HealthStatus]:
        return [
            HealthStatus(service="postgres", status="ok"),
            HealthStatus(service="redis", status="ok"),
        ]


def _fake_user() -> User:
    return User(
        id="test-user",
        username="tester",
        role=UserRole.USER,
        status=UserStatus.ACTIVE,
    )


async def test_get_status() -> None:
    app.dependency_overrides[get_current_user] = _fake_user
    app.dependency_overrides[get_status_service] = lambda: _FakeStatusService()
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.get("/api/status/")
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_status_service, None)

    data = response.json()
    assert response.status_code == 200
    assert data["code"] == 200
    assert data["data"] == [
        {"service": "postgres", "status": "ok", "details": ""},
        {"service": "redis", "status": "ok", "details": ""},
    ]
