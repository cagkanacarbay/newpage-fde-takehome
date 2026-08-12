"""The demo UI runs on localhost:3000; the API must answer its preflight."""

import asyncio
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from live_long_rnd.api.app import ApplicationConfig, create_app


@pytest.mark.e2e
def test_preflight_from_the_demo_ui_origin_is_allowed() -> None:
    async def exercise() -> httpx.Response:
        transport = ASGITransport(app=create_app())
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.options(
                "/api/chat",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                },
            )

    response = asyncio.run(exercise())
    assert response.status_code == httpx.codes.OK
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


@pytest.mark.e2e
def test_conversation_delete_preflight_is_allowed(tmp_path: Path) -> None:
    async def exercise() -> httpx.Response:
        transport = ASGITransport(
            app=create_app(config=ApplicationConfig(database_path=tmp_path / "conversations.db"))
        )
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.options(
                "/api/conversations/conversation-1",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "DELETE",
                },
            )

    response = asyncio.run(exercise())
    assert response.status_code == httpx.codes.OK
    assert "DELETE" in response.headers["access-control-allow-methods"]
