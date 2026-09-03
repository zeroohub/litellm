import pytest

from litellm.litellm_core_utils.outbound_proxy import (
    OUTBOUND_PROXY_REF_PARAM,
    OutboundProxyConfigError,
    resolve_outbound_proxy,
)
from litellm.proxy.proxy_server import ProxyConfig


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-5",
        "zai/glm-4.5",
        "deepseek/deepseek-chat",
        "moonshot/kimi-k2",
        "minimax/MiniMax-M2",
        "chatgpt/gpt-5",
    ],
)
def test_prepare_outbound_proxy_for_supported_provider(model: str) -> None:
    config = {
        "model_list": [
            {
                "model_name": "proxied",
                "litellm_params": {
                    "model": model,
                    "outbound_proxy": "socks5://user:p%40ss@127.0.0.1:1080",
                },
            }
        ]
    }

    prepared = ProxyConfig()._prepare_outbound_proxies(config)
    litellm_params = prepared["model_list"][0]["litellm_params"]
    reference = litellm_params[OUTBOUND_PROXY_REF_PARAM]

    assert "outbound_proxy" not in litellm_params
    assert "p%40ss" not in reference
    assert resolve_outbound_proxy(reference) == "socks5h://user:p%40ss@127.0.0.1:1080"


def test_prepare_outbound_proxy_supports_custom_openai_api_base() -> None:
    config = {
        "model_list": [
            {
                "model_name": "proxied",
                "litellm_params": {
                    "model": "vendor-model",
                    "custom_llm_provider": "openai",
                    "api_base": "https://models.example/v1",
                    "outbound_proxy": "http://127.0.0.1:8080",
                },
            }
        ]
    }

    prepared = ProxyConfig()._prepare_outbound_proxies(config)

    assert OUTBOUND_PROXY_REF_PARAM in prepared["model_list"][0]["litellm_params"]


@pytest.mark.parametrize("model", ["azure/gpt-5", "anthropic/claude-sonnet-4-5", "gemini/gemini-2.5-pro"])
def test_prepare_outbound_proxy_rejects_unsupported_provider(model: str) -> None:
    config = {
        "model_list": [
            {
                "model_name": "unsupported",
                "litellm_params": {"model": model, "outbound_proxy": "http://127.0.0.1:8080"},
            }
        ]
    }

    with pytest.raises(OutboundProxyConfigError, match="not supported"):
        ProxyConfig()._prepare_outbound_proxies(config)


@pytest.mark.parametrize(
    "config",
    [
        {"outbound_proxy": "http://127.0.0.1:8080", "model_list": []},
        {"litellm_settings": {"outbound_proxy": "http://127.0.0.1:8080"}, "model_list": []},
        {
            "model_list": [
                {
                    "model_name": "bad-location",
                    "outbound_proxy": "http://127.0.0.1:8080",
                    "litellm_params": {"model": "openai/gpt-5"},
                }
            ]
        },
    ],
)
def test_prepare_outbound_proxy_rejects_invalid_location(config: dict) -> None:
    with pytest.raises(OutboundProxyConfigError, match="only valid"):
        ProxyConfig()._prepare_outbound_proxies(config)


def test_prepare_outbound_proxy_rejects_environment_reference() -> None:
    config = {
        "model_list": [
            {
                "model_name": "proxied",
                "litellm_params": {
                    "model": "openai/gpt-5",
                    "outbound_proxy": "os.environ/PROXY_URL",
                },
            }
        ]
    }

    with pytest.raises(OutboundProxyConfigError, match="environment references"):
        ProxyConfig()._prepare_outbound_proxies(config)


def test_db_outbound_proxy_is_rejected() -> None:
    with pytest.raises(OutboundProxyConfigError, match=r"config\.yaml"):
        ProxyConfig()._reject_untrusted_outbound_proxy_params(
            {"model_list": [{"litellm_params": {"outbound_proxy": "http://127.0.0.1:8080"}}]}
        )


@pytest.mark.parametrize("key", ["outbound_proxy", OUTBOUND_PROXY_REF_PARAM])
def test_db_litellm_proxy_params_are_rejected(key: str) -> None:
    with pytest.raises(OutboundProxyConfigError, match=r"config\.yaml"):
        ProxyConfig()._resolve_db_litellm_param(key, "untrusted")
