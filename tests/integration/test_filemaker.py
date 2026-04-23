import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx


@pytest.mark.asyncio
async def test_filemaker_client_token_refresh(respx_mock):
    import base64
    from core.config import FileMakerSettings
    from services.filemaker import FileMakerClient

    settings = FileMakerSettings(
        host="https://fm.example.com",
        database="TestDB",
        username="user",
        password="pass",
        token_refresh_interval=1,
    )

    respx_mock.post("https://fm.example.com/fmi/data/v2/databases/TestDB/sessions").mock(
        return_value=httpx.Response(200, json={"response": {"token": "tok123"}})
    )
    respx_mock.delete(httpx.URL("https://fm.example.com/fmi/data/v2/databases/TestDB/sessions/tok123")).mock(
        return_value=httpx.Response(200, json={"response": {}})
    )

    async with FileMakerClient(settings) as client:
        assert client._token == "tok123"
