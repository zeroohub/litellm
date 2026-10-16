"""
Unit tests for the Responses-API streaming-fallback helpers added to Router
in PR #28215 (fix(router): wrap aresponses streaming iterator for mid-stream
fallbacks).

Targets the four helpers introduced on Router:
  - _extract_partial_responses_usage
  - _combine_responses_fallback_usage
  - _build_responses_continuation_input
  - _aresponses_streaming_iterator
"""

from collections.abc import Mapping, MutableMapping
from typing import Any, AsyncIterator, Final, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


from litellm import Router
from litellm.types.llms.openai import (
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)


def _make_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-test",
                },
            },
            {
                "model_name": "fallback",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "sk-test",
                },
            },
        ]
    )


def _make_completed_event(
    input_tokens: int, output_tokens: int, total_tokens: int
) -> ResponseCompletedEvent:
    response = ResponsesAPIResponse.model_construct(
        usage=ResponseAPIUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
    )
    return ResponseCompletedEvent.model_construct(
        type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
        response=response,
    )


# -------- _extract_partial_responses_usage --------


def test_extract_partial_responses_usage_native_completed():
    """Native path: completed_response carries usage → returned as-is."""
    completed = _make_completed_event(11, 7, 18)
    source = MagicMock()
    source.completed_response = completed

    usage = Router._extract_partial_responses_usage(source)
    assert usage is not None
    assert usage.input_tokens == 11
    assert usage.output_tokens == 7
    assert usage.total_tokens == 18


def test_extract_partial_responses_usage_no_completed_response():
    """Native path: no completed_response → returns None."""
    source = MagicMock()
    source.completed_response = None

    usage = Router._extract_partial_responses_usage(source)
    assert usage is None


def test_extract_partial_responses_usage_bridge_iterator_no_completed_response():
    """
    Regression for #35411: the bridge iterator
    (LiteLLMCompletionStreamingIterator) overrides __init__ without calling
    super().__init__(), so completed_response was never set until the stream
    reached RESPONSE_COMPLETED. On a mid-stream provider error (before
    completion) the fallback recovery path read source_iterator.completed_response
    and raised AttributeError, masking the real provider error and bypassing
    fallbacks. The attribute must always exist and default to None.
    """
    from litellm.responses.litellm_completion_transformation.streaming_iterator import (
        LiteLLMCompletionStreamingIterator,
    )

    wrapper = MagicMock()
    wrapper.logging_obj = MagicMock()
    iterator = LiteLLMCompletionStreamingIterator(
        model="anthropic/claude-sonnet-4-5",
        litellm_custom_stream_wrapper=wrapper,
        request_input="hi",
        responses_api_request={},
    )

    assert iterator.completed_response is None
    # No chat chunks collected yet and no completed_response → must return
    # None instead of raising AttributeError.
    assert Router._extract_partial_responses_usage(iterator) is None


# -------- _combine_responses_fallback_usage --------


def test_combine_responses_fallback_usage_sums_completed_event():
    """Partial-stream usage is summed into the fallback event's usage."""
    fallback_event = _make_completed_event(5, 3, 8)
    partial = ResponseAPIUsage(input_tokens=11, output_tokens=7, total_tokens=18)

    Router._combine_responses_fallback_usage(fallback_event, partial)

    combined = fallback_event.response.usage
    assert combined is not None
    assert combined.input_tokens == 16
    assert combined.output_tokens == 10
    assert combined.total_tokens == 26


def test_combine_responses_fallback_usage_passthrough_for_unknown_event():
    """Events that are not completed/failed/incomplete are not mutated."""
    other = MagicMock()  # not a ResponseCompletedEvent etc. → isinstance false
    partial = ResponseAPIUsage(input_tokens=1, output_tokens=1, total_tokens=2)
    Router._combine_responses_fallback_usage(other, partial)
    # No mutation expected on the unknown event — call is a no-op.


# -------- _build_responses_continuation_input --------


def test_build_responses_continuation_input_from_string():
    out = Router._build_responses_continuation_input(
        "Hello world", "partial assistant text"
    )
    assert len(out) == 3
    assert out[0]["role"] == "user"
    assert out[0]["content"][0]["text"] == "Hello world"
    assert out[1]["role"] == "developer"
    assert out[2]["role"] == "assistant"
    assert out[2]["content"][0]["text"] == "partial assistant text"


def test_build_responses_continuation_input_from_list_preserves_items():
    existing: List[Any] = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "msg1"}],
        }
    ]
    out = Router._build_responses_continuation_input(existing, "partial")
    assert len(out) == 3
    assert out[0]["content"][0]["text"] == "msg1"
    assert out[1]["role"] == "developer"
    assert out[2]["role"] == "assistant"


def test_build_responses_continuation_input_from_none():
    out = Router._build_responses_continuation_input(None, "partial")
    assert len(out) == 2
    assert out[0]["role"] == "developer"
    assert out[1]["role"] == "assistant"


# -------- _aresponses_streaming_iterator (passthrough smoke test) --------


@pytest.mark.asyncio
async def test_aresponses_streaming_iterator_passthrough():
    """
    Without MidStreamFallbackError, the wrapper yields source events
    unchanged and returns a BaseResponsesAPIStreamingIterator subclass.
    """
    from litellm.responses.streaming_iterator import (
        BaseResponsesAPIStreamingIterator,
    )

    events = [_make_completed_event(1, 1, 2)]

    class _FakeSource:
        """Minimal source iterator. Provides every attribute the wrapper
        constructor reads from source_iterator."""

        def __init__(self) -> None:
            self._i = 0
            self.completed_response = None
            self.response = MagicMock()
            self.model = "openai/gpt-4o-mini"
            self.logging_obj = MagicMock()
            self.responses_api_provider_config = MagicMock()
            self.start_time = 0.0
            self.litellm_metadata = {}
            self.custom_llm_provider = "openai"
            self.request_data = {}
            self.call_type = "aresponses"
            self._hidden_params: dict = {}

        def __aiter__(self) -> AsyncIterator[Any]:
            return self

        async def __anext__(self):
            if self._i >= len(events):
                raise StopAsyncIteration
            ev = events[self._i]
            self._i += 1
            return ev

        async def aclose(self):
            return None

    router = _make_router()
    source = _FakeSource()

    wrapper = await router._aresponses_streaming_iterator(
        source, initial_kwargs={"model": "primary"}
    )
    assert isinstance(wrapper, BaseResponsesAPIStreamingIterator)

    collected = [ev async for ev in wrapper]
    assert len(collected) == 1
    assert collected[0].type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED


# -------- _aresponses_with_streaming_fallbacks --------


@pytest.mark.asyncio
async def test_aresponses_with_streaming_fallbacks_non_streaming_passthrough():
    """Non-streaming response is returned unchanged, no wrap."""
    router = _make_router()
    plain_response = MagicMock()

    async def fake_original(**_kwargs):
        return plain_response

    with patch.object(
        router,
        "_ageneric_api_call_with_fallbacks_helper",
        new=AsyncMock(return_value=plain_response),
    ):
        out = await router._aresponses_with_streaming_fallbacks(
            original_function=fake_original,
            model="primary",
            stream=False,
        )
    assert out is plain_response


@pytest.mark.asyncio
async def test_aresponses_with_streaming_fallbacks_wraps_streaming_iterator():
    """Streaming response is wrapped via _aresponses_streaming_iterator."""
    from litellm.responses.streaming_iterator import (
        BaseResponsesAPIStreamingIterator,
    )

    router = _make_router()
    streaming_iter = MagicMock(spec=BaseResponsesAPIStreamingIterator)
    wrapped = MagicMock(spec=BaseResponsesAPIStreamingIterator)

    async def fake_original(**_kwargs):
        return streaming_iter

    with patch.object(
        router,
        "_ageneric_api_call_with_fallbacks_helper",
        new=AsyncMock(return_value=streaming_iter),
    ), patch.object(
        router,
        "_aresponses_streaming_iterator",
        new=AsyncMock(return_value=wrapped),
    ) as mock_wrap:
        out = await router._aresponses_with_streaming_fallbacks(
            original_function=fake_original,
            model="primary",
            stream=True,
        )
    assert out is wrapped
    mock_wrap.assert_awaited_once()


# -------- every fallback entry stays reachable across hops --------


def _make_three_tier_router(**router_kwargs) -> Router:
    return Router(
        model_list=[
            {"model_name": "primary", "litellm_params": {"model": "openai/primary-model", "api_key": "sk-test"}},
            {"model_name": "fb1", "litellm_params": {"model": "openai/fb1-model", "api_key": "sk-test"}},
            {"model_name": "fb2", "litellm_params": {"model": "openai/fb2-model", "api_key": "sk-test"}},
        ],
        num_retries=0,
        **router_kwargs,
    )


def _mid_stream_failure(model: str):
    import litellm
    from litellm.exceptions import MidStreamFallbackError

    return MidStreamFallbackError(
        message="stream dropped",
        model=model,
        llm_provider="openai",
        original_exception=litellm.InternalServerError(message="stream dropped", llm_provider="openai", model=model),
        is_pre_first_chunk=True,
    )


def _scripted_responses_stream(events: list, error: Exception | None = None):
    from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator

    class _ScriptedStream(BaseResponsesAPIStreamingIterator):
        def __init__(self) -> None:
            self._events = list(events)
            self._hidden_params: dict = {}
            self.completed_response = None

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._events:
                return self._events.pop(0)
            if error is not None:
                raise error
            raise StopAsyncIteration

        async def aclose(self) -> None:
            return None

    return _ScriptedStream()


def _three_tier_original(calls: list, primary_fails_pre_stream: bool):
    import litellm

    completed_event = _make_completed_event(1, 1, 2)

    async def fake_original(**kwargs):
        model = kwargs["model"]
        calls.append(model)
        if model == "openai/primary-model":
            if primary_fails_pre_stream:
                raise litellm.InternalServerError(message="primary down", llm_provider="openai", model=model)
            return _scripted_responses_stream([], _mid_stream_failure(model))
        if model == "openai/fb1-model":
            return _scripted_responses_stream([], _mid_stream_failure(model))
        return _scripted_responses_stream([completed_event])

    return fake_original, completed_event


@pytest.mark.asyncio
async def test_aresponses_pre_stream_primary_failure_then_hop_stream_failure_reaches_second_entry():
    """Regression: fallbacks=[{"primary": ["fb1", "fb2"]}]. The primary fails before streaming,
    fb1 is reached through the regular fallback chain and then fails mid-stream. Only the
    primary's stream used to be wrapped, so fb1's mid-stream failure either re-raised or
    re-tried fb1 itself; fb2 was unreachable."""
    router = _make_three_tier_router(fallbacks=[{"primary": ["fb1", "fb2"]}])
    calls: list = []
    fake_original, completed_event = _three_tier_original(calls, primary_fails_pre_stream=True)

    stream = await router._aresponses_with_streaming_fallbacks(
        original_function=fake_original, model="primary", stream=True, input="hi"
    )
    collected = [event async for event in stream]

    assert calls == ["openai/primary-model", "openai/fb1-model", "openai/fb2-model"]
    assert collected == [completed_event]


@pytest.mark.asyncio
async def test_aresponses_two_consecutive_mid_stream_failures_reach_second_entry():
    """Regression: the primary and fb1 both fail mid-stream; fb2 must still be tried."""
    router = _make_three_tier_router(fallbacks=[{"primary": ["fb1", "fb2"]}])
    calls: list = []
    fake_original, completed_event = _three_tier_original(calls, primary_fails_pre_stream=False)

    stream = await router._aresponses_with_streaming_fallbacks(
        original_function=fake_original, model="primary", stream=True, input="hi"
    )
    collected = [event async for event in stream]

    assert calls == ["openai/primary-model", "openai/fb1-model", "openai/fb2-model"]
    assert collected == [completed_event]


@pytest.mark.asyncio
async def test_aresponses_per_request_fallbacks_survive_into_hop_streams():
    """Regression: a request-level fallbacks list (key or team router_settings) is popped
    before each attempt runs, so a hop's mid-stream re-entry used to see only the router's
    own (empty) list and gave up after fb1."""
    router = _make_three_tier_router()
    calls: list = []
    fake_original, completed_event = _three_tier_original(calls, primary_fails_pre_stream=False)

    stream = await router._aresponses_with_streaming_fallbacks(
        original_function=fake_original,
        model="primary",
        stream=True,
        input="hi",
        fallbacks=[{"primary": ["fb1", "fb2"]}],
    )
    collected = [event async for event in stream]

    assert calls == ["openai/primary-model", "openai/fb1-model", "openai/fb2-model"]
    assert collected == [completed_event]


@pytest.mark.asyncio
async def test_aresponses_attempt_strips_the_controls_carrier_and_wraps_every_hop_stream():
    """Each attempt of the chain, not only the primary's, comes back wrapped for mid-stream
    failover, and the per-request controls carrier rides into the wrapper's re-entry kwargs
    without ever reaching the provider call."""
    from types import MappingProxyType

    from litellm.router_utils.fallback_event_handlers import (
        MID_STREAM_FALLBACK_CONTROLS_KEY,
        MidStreamFallbackControls,
    )

    router = _make_three_tier_router()
    completed_event = _make_completed_event(1, 1, 2)
    hop_stream = _scripted_responses_stream([completed_event])
    seen: dict = {}

    async def fake_original(**kwargs):
        seen.update(kwargs)
        return hop_stream

    controls = MidStreamFallbackControls(MappingProxyType({"fallbacks": [{"primary": ["fb1", "fb2"]}]}))
    stream = await router._ageneric_api_call_with_fallbacks_responses_attempt(
        model="fb1",
        original_generic_function=fake_original,
        stream=True,
        input="hi",
        **{MID_STREAM_FALLBACK_CONTROLS_KEY: controls},
    )
    collected = [event async for event in stream]

    assert seen["model"] == "openai/fb1-model"
    assert MID_STREAM_FALLBACK_CONTROLS_KEY not in seen
    assert "fallbacks" not in seen
    assert stream is not hop_stream
    assert collected == [completed_event]


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [429, 500])
async def test_aresponses_fallback_on_in_stream_error_event(status_code: int):
    """A retriable in-stream error event must trigger the router's mid-stream
    fallback path: the wrapper catches MidStreamFallbackError raised by the source
    iterator and yields the fallback stream instead of surfacing the error."""
    import json
    from unittest.mock import Mock

    import litellm
    from litellm.exceptions import MidStreamFallbackError
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
    from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator
    from litellm.types.llms.openai import ErrorEvent, ErrorEventError

    router = _make_router()

    error_fields: Final = (
        {"type": "tokens", "code": "rate_limit_exceeded", "message": "rate limited"}
        if status_code == 429
        else {
            "type": "server_error",
            "code": "internal_error",
            "message": "Our servers are currently overloaded. Please try again later",
        }
    )

    error_payload = {
        "type": "error",
        "error": error_fields,
    }
    sse_bytes = f"data: {json.dumps(error_payload)}\n\n".encode()

    async def mock_aiter_bytes():
        yield sse_bytes

    mock_response = Mock()
    mock_response.headers = {}
    mock_response.aiter_bytes = mock_aiter_bytes
    mock_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    mock_logging_obj.model_call_details = {"litellm_params": {}}
    mock_logging_obj.completion_start_time = None
    mock_config = Mock(spec=BaseResponsesAPIConfig)
    mock_config.transform_streaming_response.return_value = ErrorEvent(
        type=ResponsesAPIStreamEvents.ERROR,
        sequence_number=0,
        error=ErrorEventError(**error_fields),
    )

    source = ResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-5",
        responses_api_provider_config=mock_config,
        logging_obj=mock_logging_obj,
        custom_llm_provider="openai",
    )

    fallback_event = _make_completed_event(1, 1, 2)

    class _FallbackStream:
        def __init__(self) -> None:
            self._done = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._done:
                raise StopAsyncIteration
            self._done = True
            return fallback_event

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(return_value=_FallbackStream()),
    ) as mock_fallback:
        wrapped = await router._aresponses_streaming_iterator(
            response=source,
            initial_kwargs={
                "model": "primary",
                "input": "original question",
                "litellm_metadata": {"model_info": {"id": "deployment-a"}},
            },
        )
        collected = [ev async for ev in wrapped]

    assert collected == [fallback_event]
    mock_fallback.assert_awaited_once()
    raised = mock_fallback.await_args.kwargs["e"]
    assert isinstance(raised, MidStreamFallbackError)
    assert raised.status_code == status_code
    assert raised.failed_deployment_id == "deployment-a"
    expected_error_type: Final = litellm.RateLimitError if status_code == 429 else litellm.InternalServerError
    assert isinstance(raised.original_exception, expected_error_type)
    assert raised.original_exception.status_code == status_code
    assert mock_fallback.await_args.kwargs["kwargs"]["input"] == "original question"


@pytest.mark.asyncio
@pytest.mark.parametrize("client_api_key", [None, "tenant-supplied-key"])
async def test_aresponses_attempt_stamps_selected_deployment_from_stream_snapshot(client_api_key: str | None):
    from litellm import InternalServerError

    router: Final = _make_router()
    failure: Final = _mid_stream_failure("openai/gpt-4o-mini")
    selected_ids: Final[list[object]] = []
    live_model_info: Final[list[MutableMapping[str, object]]] = []

    async def original(**kwargs: object):
        model_info: Final = kwargs["model_info"]
        assert isinstance(model_info, MutableMapping)
        selected_ids.append(model_info["id"])
        live_model_info.append(model_info)
        return _scripted_responses_stream([], failure)

    stream: Final = await router._aresponses_with_streaming_fallbacks(
        original_function=original,
        model="primary",
        stream=True,
        input="hi",
        num_retries=0,
        **({"api_key": client_api_key} if client_api_key is not None else {}),
    )
    live_model_info[0]["id"] = "later-mutation"
    with pytest.raises(InternalServerError, match="stream dropped"):
        _ = [event async for event in stream]

    assert getattr(failure, "failed_deployment_id", None) == selected_ids[0]
    assert len(selected_ids) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
@pytest.mark.parametrize("existing_id", [None, "original-deployment"])
async def test_aresponses_stamps_active_metadata_without_overwriting_existing_id(
    metadata_key: str, existing_id: str | None
):
    from litellm import InternalServerError

    router: Final = _make_router()
    failure: Final = _mid_stream_failure("openai/test-model")
    if existing_id is not None:
        router._set_failed_deployment_id_on_exception(failure, {"model_info": {"id": existing_id}})
    stream: Final = await router._aresponses_streaming_iterator(
        response=_scripted_responses_stream([], failure),
        initial_kwargs={"model": "primary", metadata_key: {"model_info": {"id": "selected-deployment"}}},
    )

    with pytest.raises(InternalServerError, match="stream dropped"):
        _ = [event async for event in stream]

    assert getattr(failure, "failed_deployment_id", None) == (existing_id or "selected-deployment")


@pytest.mark.asyncio
@pytest.mark.parametrize("failures_before_success", [1, 2])
async def test_aresponses_weighted_failover_excludes_each_failed_stream(failures_before_success: int):
    router: Final = Router(
        model_list=[
            {
                "model_name": "same-group",
                "litellm_params": {"model": "openai/test-model", "api_key": "sk-test", "weight": 1},
                "model_info": {"id": deployment_id},
            }
            for deployment_id in ("account-a", "account-b", "account-c")
        ],
        num_retries=0,
        enable_weighted_failover=True,
        routing_strategy="simple-shuffle",
    )
    completed_event: Final = _make_completed_event(1, 1, 2)
    attempted_ids: Final[list[object]] = []

    async def original(**kwargs: object):
        model_info: Final = kwargs["model_info"]
        assert isinstance(model_info, Mapping)
        attempted_ids.append(model_info["id"])
        if len(attempted_ids) <= failures_before_success:
            return _scripted_responses_stream([], _mid_stream_failure("openai/test-model"))
        return _scripted_responses_stream([completed_event])

    stream: Final = await router._aresponses_with_streaming_fallbacks(
        original_function=original, model="same-group", stream=True, input="hi"
    )
    collected: Final = [event async for event in stream]

    assert collected == [completed_event]
    assert len(attempted_ids) == failures_before_success + 1
    assert len(set(attempted_ids)) == len(attempted_ids)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled, expected_attempts", [(True, 2), (False, 1)])
async def test_aresponses_failed_streams_stop_when_no_failover_is_available(enabled: bool, expected_attempts: int):
    from litellm import InternalServerError

    router: Final = Router(
        model_list=[
            {
                "model_name": "same-group",
                "litellm_params": {"model": "openai/test-model", "api_key": "sk-test", "weight": 1},
                "model_info": {"id": deployment_id},
            }
            for deployment_id in ("account-a", "account-b")
        ],
        num_retries=0,
        enable_weighted_failover=enabled,
    )
    attempted_ids: Final[list[object]] = []

    async def original(**kwargs: object):
        model_info: Final = kwargs["model_info"]
        assert isinstance(model_info, Mapping)
        attempted_ids.append(model_info["id"])
        return _scripted_responses_stream([], _mid_stream_failure("openai/test-model"))

    stream: Final = await router._aresponses_with_streaming_fallbacks(
        original_function=original, model="same-group", stream=True, input="hi"
    )
    with pytest.raises(InternalServerError, match="stream dropped"):
        _ = [event async for event in stream]

    assert len(attempted_ids) == expected_attempts
    assert len(set(attempted_ids)) == expected_attempts


@pytest.mark.asyncio
async def test_aresponses_fallback_uses_continuation_input_after_partial_content():
    """When output text was already streamed before the error, the fallback re-entry
    must carry a continuation input with the partial assistant text instead of
    retrying the original input from scratch (which would duplicate streamed content)."""
    import json
    from unittest.mock import Mock

    from litellm.exceptions import MidStreamFallbackError
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
    from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator
    from litellm.types.llms.openai import ErrorEvent, ErrorEventError

    router = _make_router()

    events = [
        {"type": "response.output_text.delta", "delta": "partial answer"},
        {"type": "error", "error": {"type": "server_error", "code": "internal_error", "message": "boom"}},
    ]
    sse_payload = b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events)

    async def mock_aiter_bytes():
        yield sse_payload

    mock_response = Mock()
    mock_response.headers = {}
    mock_response.aiter_bytes = mock_aiter_bytes
    mock_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    mock_logging_obj.model_call_details = {"litellm_params": {}}
    mock_logging_obj.completion_start_time = None
    mock_config = Mock(spec=BaseResponsesAPIConfig)

    def transform(model, parsed_chunk, logging_obj):
        if parsed_chunk.get("type") == "error":
            return ErrorEvent(
                type=ResponsesAPIStreamEvents.ERROR,
                sequence_number=0,
                error=ErrorEventError(**parsed_chunk["error"]),
            )
        delta_event = Mock()
        delta_event.type = ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
        delta_event.delta = parsed_chunk["delta"]
        return delta_event

    mock_config.transform_streaming_response.side_effect = transform

    source = ResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-5",
        responses_api_provider_config=mock_config,
        logging_obj=mock_logging_obj,
        custom_llm_provider="openai",
    )

    fallback_event = _make_completed_event(1, 1, 2)

    class _FallbackStream:
        def __init__(self) -> None:
            self._done = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._done:
                raise StopAsyncIteration
            self._done = True
            return fallback_event

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(return_value=_FallbackStream()),
    ) as mock_fallback:
        wrapped = await router._aresponses_streaming_iterator(
            response=source,
            initial_kwargs={"model": "primary", "input": "original question"},
        )
        collected = [ev async for ev in wrapped]

    assert collected[-1] == fallback_event
    raised = mock_fallback.await_args.kwargs["e"]
    assert isinstance(raised, MidStreamFallbackError)
    assert raised.is_pre_first_chunk is False
    assert raised.generated_content == "partial answer"
    continuation = mock_fallback.await_args.kwargs["kwargs"]["input"]
    assert isinstance(continuation, list)
    assert continuation[0]["content"][0]["text"] == "original question"
    assert continuation[-2]["role"] == "developer"
    assert continuation[-1]["role"] == "assistant"
    assert continuation[-1]["content"][0]["text"] == "partial answer"


@pytest.mark.asyncio
async def test_aresponses_client_error_event_skips_fallback():
    """A 400-mapped in-stream error (raised as APIError, not MidStreamFallbackError)
    must surface to the caller without invoking the router's fallback path."""
    import litellm

    router = _make_router()

    class _ClientErrorSource:
        completed_response = None

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise litellm.APIError(
                status_code=400,
                message="bad request",
                llm_provider="openai",
                model="gpt-5",
            )

    wrapped = await router._aresponses_streaming_iterator(
        response=_ClientErrorSource(),
        initial_kwargs={"model": "primary"},
    )

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(),
    ) as mock_fallback:
        with pytest.raises(litellm.APIError) as exc_info:
            async for _ in wrapped:
                pass

    assert exc_info.value.status_code == 400
    mock_fallback.assert_not_awaited()
