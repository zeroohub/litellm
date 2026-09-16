import base64
from io import BufferedReader, BytesIO
from typing import Final, TypeAlias, cast

from httpx._types import RequestFiles

from litellm.exceptions import AuthenticationError
from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.openai.image_edit.transformation import OpenAIImageEditConfig
from litellm.llms.openai.image_generation.gpt_transformation import GPTImageGenerationConfig
from litellm.types.llms.openai import AllMessageValues, OpenAIImageGenerationOptionalParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes

from ..authenticator import resolve_chatgpt_credentials
from ..common_utils import ChatGPTAuthError, get_chatgpt_api_base, get_chatgpt_default_headers

ImageInput: TypeAlias = bytes | BytesIO | BufferedReader
RequestData: TypeAlias = dict[str, object]


def _endpoint_url(api_base: str | None, endpoint: str) -> str:
    base_url: Final = (api_base or get_chatgpt_api_base()).rstrip("/")
    known_suffixes: Final = ("/images/generations", "/images/edits")
    normalized_base: Final = next(
        (base_url[: -len(suffix)] for suffix in known_suffixes if base_url.endswith(suffix)),
        base_url,
    )
    return f"{normalized_base}/images/{endpoint}"


def _auth_headers(configured_api_key: str | None, model: str, headers: RequestData) -> RequestData:
    try:
        credentials: Final = resolve_chatgpt_credentials(configured_api_key)
    except ChatGPTAuthError as exc:
        raise AuthenticationError(model=model, llm_provider="chatgpt", message=str(exc))
    default_headers: Final[RequestData] = cast(
        RequestData,
        get_chatgpt_default_headers(credentials.access_token, credentials.account_id),
    )
    return {**default_headers, "accept": "application/json", **headers}


def _read_image_bytes(image: ImageInput) -> bytes:
    if isinstance(image, bytes):
        return image
    current_position: Final = image.tell()
    image.seek(0)
    data: Final = image.read()
    image.seek(current_position)
    return data


def _normalize_images(image: object) -> tuple[ImageInput, ...]:
    image_values: Final[tuple[object, ...]] = tuple(cast(list[object], image)) if isinstance(image, list) else (image,)
    if not image_values or image_values == (None,):
        raise ValueError("ChatGPT image edit requires at least one image")
    if not all(isinstance(item, (bytes, BytesIO, BufferedReader)) for item in image_values):
        raise ValueError("Unsupported image type for ChatGPT image edit")
    return cast(tuple[ImageInput, ...], image_values)


class ChatGPTImageGenerationConfig(GPTImageGenerationConfig):
    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        return ["background", "n", "quality", "size"]

    def map_openai_params(
        self,
        non_default_params: RequestData,
        optional_params: RequestData,
        model: str,
        drop_params: bool,
    ) -> RequestData:
        supported_params: Final = self.get_supported_openai_params(model)
        unsupported_params: Final = tuple(param for param in non_default_params if param not in supported_params)
        if unsupported_params and not drop_params:
            raise ValueError(
                f"Parameters {', '.join(unsupported_params)} are not supported for model {model}. "
                "Set drop_params=True to drop unsupported parameters."
            )
        return {
            **optional_params,
            **{key: value for key, value in non_default_params.items() if key in supported_params},
        }

    def validate_environment(
        self,
        headers: RequestData,
        model: str,
        messages: list[AllMessageValues],
        optional_params: RequestData,
        litellm_params: RequestData,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> RequestData:
        return _auth_headers(api_key, model, headers)

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: RequestData,
        litellm_params: RequestData,
        stream: bool | None = None,
    ) -> str:
        return _endpoint_url(api_base, "generations")

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: RequestData,
        litellm_params: RequestData,
        headers: RequestData,
    ) -> RequestData:
        return {"model": model, "prompt": prompt, **optional_params}


class ChatGPTImageEditConfig(OpenAIImageEditConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:
        return ["background", "input_fidelity", "n", "quality", "size"]

    def validate_environment(
        self,
        headers: RequestData,
        model: str,
        api_key: str | None = None,
        litellm_params: RequestData | None = None,
        api_base: str | None = None,
    ) -> RequestData:
        return _auth_headers(api_key, model, headers)

    def get_complete_url(self, model: str, api_base: str | None, litellm_params: RequestData) -> str:
        return _endpoint_url(api_base, "edits")

    def transform_image_edit_request(
        self,
        model: str,
        prompt: str | None,
        image: FileTypes | None,
        image_edit_optional_request_params: RequestData,
        litellm_params: GenericLiteLLMParams,
        headers: RequestData,
    ) -> tuple[RequestData, RequestFiles]:
        image_list: Final = _normalize_images(image)
        images: Final = tuple(
            {
                "image_url": (
                    f"data:{ImageEditRequestUtils.get_image_content_type(item)};base64,"
                    f"{base64.b64encode(_read_image_bytes(item)).decode('ascii')}"
                )
            }
            for item in image_list
        )
        request: Final[RequestData] = {
            "model": model,
            "prompt": prompt or "",
            "images": list(images),
            **image_edit_optional_request_params,
        }
        return request, cast(RequestFiles, [])

    def use_multipart_form_data(self) -> bool:
        return False
