import httpx
import pytest

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.outbound_proxy import (
    register_outbound_proxy,
    set_current_outbound_proxy_reference,
)
from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport
from litellm.llms.custom_httpx.http_handler import _get_httpx_client, get_async_httpx_client


def _pool_proxy(transport: object) -> tuple[str, str, int] | None:
    url = getattr(getattr(transport, "_pool", None), "_proxy_url", None)
    if url is None:
        return None
    return url.scheme.decode(), url.host.decode(), url.port


def test_sync_clients_are_isolated_by_deployment_proxy() -> None:
    previous_cache = litellm.in_memory_llm_clients_cache
    litellm.in_memory_llm_clients_cache = LLMClientCache()
    first_ref = register_outbound_proxy("http://first:secret@127.0.0.1:8080")
    second_ref = register_outbound_proxy("http://second:secret@127.0.0.1:8081")
    try:
        set_current_outbound_proxy_reference(first_ref)
        first = _get_httpx_client()
        set_current_outbound_proxy_reference(second_ref)
        second = _get_httpx_client()
        set_current_outbound_proxy_reference(None)
        direct = _get_httpx_client()

        assert first is not second
        assert first is not direct
        assert second is not direct
        assert _pool_proxy(first.client._transport) == ("http", "127.0.0.1", 8080)
        assert _pool_proxy(second.client._transport) == ("http", "127.0.0.1", 8081)
        assert _pool_proxy(direct.client._transport) is None
        assert all("secret" not in key for key in litellm.in_memory_llm_clients_cache.cache_dict)
    finally:
        set_current_outbound_proxy_reference(None)
        litellm.in_memory_llm_clients_cache = previous_cache


@pytest.mark.asyncio
async def test_async_proxy_uses_httpx_instead_of_aiohttp() -> None:
    previous_cache = litellm.in_memory_llm_clients_cache
    litellm.in_memory_llm_clients_cache = LLMClientCache()
    reference = register_outbound_proxy("http://127.0.0.1:8080")
    try:
        set_current_outbound_proxy_reference(reference)
        handler = get_async_httpx_client(llm_provider=litellm.LlmProviders.DEEPSEEK)
        assert isinstance(handler.client._transport, httpx.AsyncHTTPTransport)
        assert not isinstance(handler.client._transport, LiteLLMAiohttpTransport)
    finally:
        set_current_outbound_proxy_reference(None)
        litellm.in_memory_llm_clients_cache = previous_cache
