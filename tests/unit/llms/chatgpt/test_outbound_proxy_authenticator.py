from pathlib import Path

from litellm.litellm_core_utils.outbound_proxy import (
    register_outbound_proxy,
    set_current_outbound_proxy_reference,
)
from litellm.llms.chatgpt.authenticator import Authenticator
from litellm.llms.custom_httpx.http_handler import _get_httpx_client


def test_chatgpt_authentication_client_uses_current_deployment_proxy(tmp_path: Path) -> None:
    reference = register_outbound_proxy("http://127.0.0.1:8080")
    try:
        set_current_outbound_proxy_reference(reference)
        Authenticator(auth_file=str(tmp_path / "auth.json"))
        client = _get_httpx_client()

        assert client.outbound_proxy_ref == reference
    finally:
        set_current_outbound_proxy_reference(None)
