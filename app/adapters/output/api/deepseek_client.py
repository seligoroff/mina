"""Адаптер для DeepSeek API."""

from typing import Any, Dict, Optional
import json

import requests

from app.application.ports import ILLMProtocolClient
from app.domain.exceptions import ProtocolClientError
from app.domain.models.protocol import ProtocolRequest, ProtocolResponse


class DeepSeekProtocolClient(ILLMProtocolClient):
    """Реализация порта для DeepSeek API."""

    DEFAULT_BASE_URL = "https://api.deepseek.com/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        http_client: Optional[Any] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 300,
    ) -> None:
        if not api_key:
            raise ValueError("DeepSeek API key is required")

        self._api_key = api_key
        self._http_client = http_client or requests
        self._base_url = base_url
        self._timeout = timeout

    def generate_protocol(self, request: ProtocolRequest) -> ProtocolResponse:
        payload: Dict[str, Any] = {
            "model": request.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": request.render_prompt(),
                }
            ],
            "temperature": request.config.temperature,
        }

        if request.config.extra_params:
            payload.update(request.config.extra_params)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = self._http_client.post(
                self._base_url,
                json=payload,
                headers=headers,
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise ProtocolClientError(f"Ошибка при отправке запроса в DeepSeek: {exc}") from exc

        if not response.ok:
            try:
                error_payload = response.json()
                error_detail = json.dumps(error_payload, ensure_ascii=False)
            except ValueError:
                error_detail = response.text
            raise ProtocolClientError(
                f"DeepSeek вернул ошибку {response.status_code}: {error_detail}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise ProtocolClientError("Некорректный JSON-ответ от DeepSeek") from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProtocolClientError("Ответ DeepSeek не содержит контент") from exc

        return ProtocolResponse(content=content, provider_raw=data)




