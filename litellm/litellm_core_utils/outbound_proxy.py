import hashlib
import hmac
from contextvars import ContextVar
from importlib import import_module
from secrets import token_bytes
from threading import RLock
from typing import Final
from urllib.parse import SplitResult, urlsplit, urlunsplit

OUTBOUND_PROXY_PARAM: Final = "outbound_proxy"
OUTBOUND_PROXY_REF_PARAM: Final = "_outbound_proxy_ref"
SUPPORTED_OUTBOUND_PROXY_PROVIDERS: Final = frozenset({"openai", "zai", "deepseek", "moonshot", "minimax", "chatgpt"})
_SUPPORTED_PROXY_SCHEMES: Final = frozenset({"http", "https", "socks5", "socks5h"})


class OutboundProxyConfigError(ValueError):
    pass


class _OutboundProxyRegistry:
    def __init__(self) -> None:
        self._proxies: dict[str, str] = {}
        self._lock: Final = RLock()

    def register(self, proxy_url: str) -> str:
        reference: Final = hmac.new(_REFERENCE_SECRET, proxy_url.encode("utf-8"), hashlib.sha256).hexdigest()
        with self._lock:
            self._proxies[reference] = proxy_url
        return reference

    def resolve(self, reference: str) -> str:
        with self._lock:
            proxy_url: Final = self._proxies.get(reference)
        if proxy_url is None:
            raise OutboundProxyConfigError("The configured outbound proxy reference is unavailable")
        return proxy_url


_REFERENCE_SECRET: Final = token_bytes(32)
_REGISTRY: Final = _OutboundProxyRegistry()
_CURRENT_OUTBOUND_PROXY_REF: Final[ContextVar[str | None]] = ContextVar("litellm_outbound_proxy_ref", default=None)


def _validate_port(parsed: SplitResult) -> int:
    try:
        port: Final = parsed.port
    except ValueError as exc:
        raise OutboundProxyConfigError("outbound_proxy has an invalid port") from exc
    if port is None:
        raise OutboundProxyConfigError("outbound_proxy must include an explicit port")
    return port


def validate_outbound_proxy_url(raw_proxy_url: object) -> str:
    if not isinstance(raw_proxy_url, str) or not raw_proxy_url.strip():
        raise OutboundProxyConfigError("outbound_proxy must be a non-empty URL")
    proxy_url: Final = raw_proxy_url.strip()
    if proxy_url.startswith("os.environ/"):
        raise OutboundProxyConfigError("outbound_proxy does not support environment references")

    parsed: Final = urlsplit(proxy_url)
    scheme: Final = parsed.scheme.lower()
    if scheme not in _SUPPORTED_PROXY_SCHEMES:
        raise OutboundProxyConfigError("outbound_proxy must use http, https, socks5, or socks5h")
    if parsed.hostname is None:
        raise OutboundProxyConfigError("outbound_proxy must include a hostname")
    _validate_port(parsed)
    if parsed.path or parsed.query or parsed.fragment:
        raise OutboundProxyConfigError("outbound_proxy must not include a path, query string, or fragment")
    if scheme in {"socks5", "socks5h"}:
        try:
            import_module("socksio")
        except ImportError as exc:
            raise OutboundProxyConfigError(
                "outbound_proxy SOCKS support requires the 'socksio' package from 'httpx[socks]'"
            ) from exc

    normalized_scheme: Final = "socks5h" if scheme == "socks5" else scheme
    return urlunsplit((normalized_scheme, parsed.netloc, "", "", ""))


def register_outbound_proxy(raw_proxy_url: object) -> str:
    return _REGISTRY.register(validate_outbound_proxy_url(raw_proxy_url))


def resolve_outbound_proxy(reference: object) -> str | None:
    if reference is None:
        return None
    if not isinstance(reference, str) or not reference:
        raise OutboundProxyConfigError("The configured outbound proxy reference is invalid")
    return _REGISTRY.resolve(reference)


def outbound_proxy_fingerprint(reference: object) -> str:
    if reference is None:
        return "direct"
    if not isinstance(reference, str) or not reference:
        raise OutboundProxyConfigError("The configured outbound proxy reference is invalid")
    _REGISTRY.resolve(reference)
    return reference


def set_current_outbound_proxy_reference(reference: object) -> None:
    if reference is not None and not isinstance(reference, str):
        raise OutboundProxyConfigError("The configured outbound proxy reference is invalid")
    _CURRENT_OUTBOUND_PROXY_REF.set(reference)


def get_current_outbound_proxy_reference() -> str | None:
    return _CURRENT_OUTBOUND_PROXY_REF.get()
