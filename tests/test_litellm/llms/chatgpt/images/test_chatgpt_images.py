import base64
import json
import time
from io import BytesIO
from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import ImageResponse

_PNG: Final = b"\x89PNG\r\n\x1a\n" + b"test-image"
_JPEG: Final = b"\xff\xd8\xff\xe0" + b"test-image"


def _write_auth_file(tmp_path) -> str:
    auth_file = tmp_path / "account" / "auth.js"
    auth_file.parent.mkdir()
    auth_file.write_text(
        json.dumps(
            {
                "access_token": "account-token",
                "expires_at": time.time() + 3600,
                "account_id": "account-id",
            }
        )
    )
    return str(auth_file)


class _SyncCapturingClient(HTTPHandler):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[dict] = []

    def post(self, url: str, headers=None, json=None, **kwargs) -> httpx.Response:
        self.requests.append({"url": url, "headers": dict(headers or {}), "json": json, **kwargs})
        return httpx.Response(
            200,
            json={"created": 123, "data": [{"b64_json": "generated-image"}]},
            request=httpx.Request("POST", url),
        )


class _AsyncCapturingClient(AsyncHTTPHandler):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[dict] = []

    async def post(self, url: str, headers=None, json=None, **kwargs) -> httpx.Response:
        self.requests.append({"url": url, "headers": dict(headers or {}), "json": json, **kwargs})
        return httpx.Response(
            200,
            json={"created": 456, "data": [{"b64_json": "edited-image"}]},
            request=httpx.Request("POST", url),
        )


def test_chatgpt_image_generation_uses_codex_json_endpoint(tmp_path):
    client = _SyncCapturingClient()
    response = litellm.image_generation(
        model="chatgpt/gpt-image-1",
        prompt="draw a lighthouse",
        api_key=_write_auth_file(tmp_path),
        api_base="https://example.test/backend-api/codex/",
        background="transparent",
        n=2,
        quality="high",
        size="1024x1024",
        client=client,
    )

    assert isinstance(response, ImageResponse)
    assert response.created == 123
    assert response.data[0].b64_json == "generated-image"
    assert client.requests == [
        {
            "url": "https://example.test/backend-api/codex/images/generations",
            "headers": {
                "Authorization": "Bearer account-token",
                "content-type": "application/json",
                "accept": "application/json",
                "originator": client.requests[0]["headers"]["originator"],
                "user-agent": client.requests[0]["headers"]["user-agent"],
                "ChatGPT-Account-Id": "account-id",
            },
            "json": {
                "model": "gpt-image-1",
                "prompt": "draw a lighthouse",
                "background": "transparent",
                "n": 2,
                "quality": "high",
                "size": "1024x1024",
            },
            "timeout": 600,
        }
    ]


@pytest.mark.asyncio
async def test_chatgpt_async_image_generation_uses_same_auth_and_json_transform(tmp_path):
    client = _AsyncCapturingClient()
    response = await litellm.aimage_generation(
        model="chatgpt/gpt-image-1",
        prompt="draw a harbor",
        api_key=_write_auth_file(tmp_path),
        api_base="https://example.test/backend-api/codex",
        n=1,
        client=client,
    )

    assert response.data[0].b64_json == "edited-image"
    assert client.requests[0]["url"] == "https://example.test/backend-api/codex/images/generations"
    assert client.requests[0]["headers"]["Authorization"] == "Bearer account-token"
    assert client.requests[0]["headers"]["ChatGPT-Account-Id"] == "account-id"
    assert client.requests[0]["json"] == {
        "model": "gpt-image-1",
        "prompt": "draw a harbor",
        "n": 1,
    }
    await client.close()


@pytest.mark.asyncio
async def test_chatgpt_image_edit_uses_json_data_urls_and_preserves_multiple_images(tmp_path):
    client = _AsyncCapturingClient()
    response = await litellm.aimage_edit(
        model="chatgpt/gpt-image-1",
        prompt="combine these images",
        image=[BytesIO(_PNG), BytesIO(_JPEG)],
        api_key=_write_auth_file(tmp_path),
        api_base="https://example.test/backend-api/codex/images/generations",
        quality="high",
        client=client,
    )

    assert response.data[0].b64_json == "edited-image"
    assert len(client.requests) == 1
    request = client.requests[0]
    assert request["url"] == "https://example.test/backend-api/codex/images/edits"
    assert request["headers"]["Authorization"] == "Bearer account-token"
    assert request["headers"]["ChatGPT-Account-Id"] == "account-id"
    assert request["json"] == {
        "model": "gpt-image-1",
        "prompt": "combine these images",
        "images": [
            {"image_url": f"data:image/png;base64,{base64.b64encode(_PNG).decode('ascii')}"},
            {"image_url": f"data:image/jpeg;base64,{base64.b64encode(_JPEG).decode('ascii')}"},
        ],
        "quality": "high",
    }
    assert "files" not in request
    await client.close()


@pytest.mark.parametrize("drop_params", [False, True])
def test_chatgpt_image_edit_mask_follows_drop_params_contract(tmp_path, drop_params):
    client = _SyncCapturingClient()
    call = lambda: litellm.image_edit(
        model="chatgpt/gpt-image-1",
        prompt="edit",
        image=BytesIO(_PNG),
        mask="mask.png",
        api_key=_write_auth_file(tmp_path),
        drop_params=drop_params,
        client=client,
    )

    if not drop_params:
        with pytest.raises(litellm.UnsupportedParamsError):
            call()
        assert client.requests == []
        return

    response = call()
    assert response.data[0].b64_json == "generated-image"
    assert "mask" not in client.requests[0]["json"]
