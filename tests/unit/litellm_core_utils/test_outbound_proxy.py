import pytest

from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
from litellm.litellm_core_utils.outbound_proxy import (
    OutboundProxyConfigError,
    register_outbound_proxy,
    resolve_outbound_proxy,
    set_current_outbound_proxy_reference,
    validate_outbound_proxy_url,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://proxy.internal:8080", "http://proxy.internal:8080"),
        ("https://user:p%40ss@proxy.internal:8443", "https://user:p%40ss@proxy.internal:8443"),
        ("socks5://127.0.0.1:1080", "socks5h://127.0.0.1:1080"),
        ("socks5h://user:pass@127.0.0.1:1080", "socks5h://user:pass@127.0.0.1:1080"),
    ],
)
def test_validate_outbound_proxy_url(raw: str, expected: str) -> None:
    assert validate_outbound_proxy_url(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "proxy.internal:8080",
        "http://proxy.internal",
        "http://proxy.internal:8080/path",
        "http://proxy.internal:8080?query=1",
        "http://proxy.internal:8080#fragment",
        "os.environ/OUTBOUND_PROXY",
    ],
)
def test_validate_outbound_proxy_url_rejects_invalid_values(raw: str) -> None:
    with pytest.raises(OutboundProxyConfigError):
        validate_outbound_proxy_url(raw)


def test_registry_uses_opaque_reference_and_fails_closed() -> None:
    proxy_url = "http://user:secret@proxy.internal:8080"
    reference = register_outbound_proxy(proxy_url)

    assert proxy_url not in reference
    assert "secret" not in reference
    assert resolve_outbound_proxy(reference) == proxy_url
    with pytest.raises(OutboundProxyConfigError):
        resolve_outbound_proxy("missing-reference")


def test_runtime_outbound_proxy_is_rejected() -> None:
    set_current_outbound_proxy_reference(None)
    with pytest.raises(ValueError, match=r"config\.yaml"):
        get_litellm_params(outbound_proxy="http://proxy.internal:8080")


def test_runtime_internal_proxy_reference_is_rejected() -> None:
    set_current_outbound_proxy_reference(None)
    with pytest.raises(ValueError, match="reserved"):
        get_litellm_params(_outbound_proxy_ref=register_outbound_proxy("http://127.0.0.1:8080"))
