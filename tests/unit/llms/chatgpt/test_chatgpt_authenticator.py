import base64
import json
import time
from unittest.mock import mock_open, patch

import pytest

from litellm.llms.chatgpt.authenticator import Authenticator
from litellm.llms.chatgpt.chat.transformation import ChatGPTConfig
from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams


def _make_jwt(payload: dict) -> str:
    header = {"alg": "none", "typ": "JWT"}

    def _b64(obj: dict) -> str:
        raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")

    return f"{_b64(header)}.{_b64(payload)}."


class TestChatGPTAuthenticator:
    @pytest.fixture
    def authenticator(self):
        with patch("os.path.exists", return_value=True):
            return Authenticator()

    def test_get_access_token_from_file(self, authenticator):
        future_time = time.time() + 3600
        auth_data = json.dumps({"access_token": "token-123", "expires_at": future_time})

        with patch("builtins.open", mock_open(read_data=auth_data)):
            token = authenticator.get_access_token()
            assert token == "token-123"

    def test_get_access_token_refresh(self, authenticator):
        past_time = time.time() - 10
        auth_data = json.dumps(
            {
                "access_token": "token-old",
                "refresh_token": "refresh-123",
                "expires_at": past_time,
            }
        )
        refreshed = {
            "access_token": "token-new",
            "refresh_token": "refresh-123",
            "id_token": "id-123",
        }

        with (
            patch("builtins.open", mock_open(read_data=auth_data)),
            patch.object(authenticator, "_refresh_tokens", return_value=refreshed),
        ):
            token = authenticator.get_access_token()
            assert token == "token-new"

    def test_get_account_id_from_id_token(self, authenticator):
        id_token = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-123"}})
        auth_data = json.dumps({"id_token": id_token})

        with (
            patch("builtins.open", mock_open(read_data=auth_data)),
            patch.object(authenticator, "_write_auth_file") as mock_write,
        ):
            account_id = authenticator.get_account_id()
            assert account_id == "acct-123"
            mock_write.assert_called_once()
            assert mock_write.call_args[0][0]["account_id"] == "acct-123"

    def test_relative_auth_file_resolves_under_token_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))

        authenticator = Authenticator(auth_file="account-a/auth.json")

        assert authenticator.auth_file == str(tmp_path / "account-a" / "auth.json")
        assert authenticator.token_dir == str(tmp_path / "account-a")

    def test_absolute_auth_file_is_used_directly(self, tmp_path, monkeypatch):
        default_token_dir = tmp_path / "default"
        selected_auth_file = tmp_path / "account-a" / "auth.json"
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(default_token_dir))

        authenticator = Authenticator(auth_file=str(selected_auth_file))

        assert authenticator.auth_file == str(selected_auth_file)
        assert authenticator.token_dir == str(selected_auth_file.parent)

    def test_chat_config_uses_selected_auth_file_and_matching_account(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
        access_token = _make_jwt(
            {
                "exp": time.time() + 3600,
                "https://api.openai.com/auth": {"chatgpt_account_id": "acct-a"},
            }
        )
        auth_file = tmp_path / "account-a" / "auth.json"
        auth_file.parent.mkdir()
        auth_file.write_text(json.dumps({"access_token": access_token, "expires_at": time.time() + 3600}))
        config = ChatGPTConfig()

        _, dynamic_api_key, _ = config._get_openai_compatible_provider_info(
            model="gpt-5.4",
            api_base=None,
            api_key="account-a/auth.json",
            custom_llm_provider="chatgpt",
        )
        headers = config.validate_environment(
            headers={},
            model="gpt-5.4",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=dynamic_api_key,
        )

        assert dynamic_api_key == access_token
        assert headers["Authorization"] == f"Bearer {access_token}"
        assert headers["ChatGPT-Account-Id"] == "acct-a"

    def test_chat_provider_detection_does_not_trigger_default_authentication(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
        config = ChatGPTConfig()

        _, dynamic_api_key, _ = config._get_openai_compatible_provider_info(
            model="gpt-5.4",
            api_base=None,
            api_key=None,
            custom_llm_provider="chatgpt",
        )

        assert dynamic_api_key is None
        assert not (tmp_path / "auth.json").exists()

    def test_responses_config_accepts_resolved_access_token(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
        access_token = _make_jwt(
            {
                "exp": time.time() + 3600,
                "https://api.openai.com/auth": {"chatgpt_account_id": "acct-a"},
            }
        )
        config = ChatGPTResponsesAPIConfig()

        headers = config.validate_environment(
            headers={},
            model="gpt-5.4",
            litellm_params=GenericLiteLLMParams(api_key=access_token),
        )

        assert headers["Authorization"] == f"Bearer {access_token}"
        assert headers["ChatGPT-Account-Id"] == "acct-a"
        assert not (tmp_path / access_token).exists()

    def test_responses_config_keeps_accounts_isolated(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
        account_tokens = {
            "account-a": _make_jwt({"exp": time.time() + 3600}),
            "account-b": _make_jwt({"exp": time.time() + 3600}),
        }
        for account_name, access_token in account_tokens.items():
            auth_file = tmp_path / account_name / "auth.json"
            auth_file.parent.mkdir()
            auth_file.write_text(
                json.dumps(
                    {
                        "access_token": access_token,
                        "expires_at": time.time() + 3600,
                        "account_id": f"acct-{account_name[-1]}",
                    }
                )
            )
        config = ChatGPTResponsesAPIConfig()

        account_a_headers = config.validate_environment(
            headers={},
            model="gpt-5.4",
            litellm_params=GenericLiteLLMParams(api_key="account-a/auth.json"),
        )
        account_b_headers = config.validate_environment(
            headers={},
            model="gpt-5.4",
            litellm_params=GenericLiteLLMParams(api_key="account-b/auth.json"),
        )

        assert account_a_headers["Authorization"] == f"Bearer {account_tokens['account-a']}"
        assert account_a_headers["ChatGPT-Account-Id"] == "acct-a"
        assert account_b_headers["Authorization"] == f"Bearer {account_tokens['account-b']}"
        assert account_b_headers["ChatGPT-Account-Id"] == "acct-b"
