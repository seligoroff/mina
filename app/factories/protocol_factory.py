"""Фабрики для компонентов протокола."""

from typing import Optional, Dict, Any

from app.domain.models.protocol import ProtocolConfig
from app.application.services import ProtocolService
from app.application.ports import ILLMProtocolClient
from app.adapters.output.api import DeepSeekProtocolClient

DEFAULT_PROVIDER = "deepseek"


def _create_deepseek_client(config: ProtocolConfig, dependencies: Optional[Dict[str, Any]] = None) -> ILLMProtocolClient:
    deps = dependencies or {}
    http_client = deps.get("http_client")
    base_url = deps.get("base_url", DeepSeekProtocolClient.DEFAULT_BASE_URL)
    timeout = deps.get("timeout", 300)

    return DeepSeekProtocolClient(
        api_key=config.api_key,
        http_client=http_client,
        base_url=base_url,
        timeout=timeout,
    )


PROVIDER_FACTORIES = {
    "deepseek": _create_deepseek_client,
}


def create_protocol_client(
    config: ProtocolConfig,
    dependencies: Optional[Dict[str, Any]] = None,
) -> ILLMProtocolClient:
    """Создаёт клиента LLM на основе настроек провайдера."""
    provider = config.provider or DEFAULT_PROVIDER
    factory = PROVIDER_FACTORIES.get(provider.lower())
    if not factory:
        raise ValueError(f"Неизвестный провайдер протокола: {provider}")
    return factory(config, dependencies)


def create_protocol_service(client: ILLMProtocolClient) -> ProtocolService:
    """Создаёт ProtocolService."""
    return ProtocolService(client=client)





