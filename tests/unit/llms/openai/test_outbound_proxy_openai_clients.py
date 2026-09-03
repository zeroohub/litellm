import httpx

import litellm
from litellm.litellm_core_utils.outbound_proxy import (
    register_outbound_proxy,
    set_current_outbound_proxy_reference,
)
from litellm.llms.openai.common_utils import BaseOpenAILLM


def _pool_proxy(transport: object) -> tuple[str, str, int] | None:
    url = getattr(getattr(transport, "_pool", None), "_proxy_url", None)
    if url is None:
        return None
    return url.scheme.decode(), url.host.decode(), url.port


def test_openai_sync_client_uses_deployment_proxy_without_reusing_global_session() -> None:
    direct_client = httpx.Client()
    previous_client = litellm.client_session
    reference = register_outbound_proxy("https://user:secret@127.0.0.1:8443")
    litellm.client_session = direct_client
    try:
        set_current_outbound_proxy_reference(reference)
        proxied_client = BaseOpenAILLM._get_sync_http_client()
        assert proxied_client is not None
        assert proxied_client is not direct_client
        assert _pool_proxy(proxied_client._transport) == ("https", "127.0.0.1", 8443)
        proxied_client.close()
    finally:
        set_current_outbound_proxy_reference(None)
        litellm.client_session = previous_client
        direct_client.close()


def test_openai_client_cache_key_is_proxy_isolated_and_credential_safe() -> None:
    first_ref = register_outbound_proxy("http://first:secret@127.0.0.1:8080")
    second_ref = register_outbound_proxy("http://second:secret@127.0.0.1:8081")
    params = {"api_key": "sk-test", "is_async": False}
    try:
        set_current_outbound_proxy_reference(first_ref)
        first_key = BaseOpenAILLM.get_openai_client_cache_key(params, "openai")
        set_current_outbound_proxy_reference(second_ref)
        second_key = BaseOpenAILLM.get_openai_client_cache_key(params, "openai")
        set_current_outbound_proxy_reference(None)
        direct_key = BaseOpenAILLM.get_openai_client_cache_key(params, "openai")

        assert len({first_key, second_key, direct_key}) == 3
        assert "first:secret" not in first_key
        assert "127.0.0.1:8080" not in first_key
    finally:
        set_current_outbound_proxy_reference(None)
