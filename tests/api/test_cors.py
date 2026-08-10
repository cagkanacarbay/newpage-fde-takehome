"""The demo UI runs on localhost:3000; the API must answer its preflight."""

import asyncio

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from live_long_rnd.api.app import create_app


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
