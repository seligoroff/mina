"""Тесты для protocol_factory."""

import pytest

from app.domain.models.protocol import ProtocolConfig
from app.factories.protocol_factory import create_protocol_client, create_protocol_service
from app.application.services import ProtocolService


class DummyClient:
    def __init__(self, config):
        self.config = config


def test_create_protocol_client_with_known_provider(monkeypatch):
    config = ProtocolConfig(provider="deepseek", api_key="key")

    factory_called = {}

    def fake_factory(cfg, deps):
        factory_called["config"] = cfg
        factory_called["deps"] = deps
        return DummyClient(cfg)

    factories = create_protocol_client.__globals__["PROVIDER_FACTORIES"].copy()
    factories["deepseek"] = fake_factory
    monkeypatch.setitem(create_protocol_client.__globals__, "PROVIDER_FACTORIES", factories)

    client = create_protocol_client(config, dependencies={"timeout": 100})

    assert isinstance(client, DummyClient)
    assert factory_called["config"] is config
    assert factory_called["deps"]["timeout"] == 100


def test_create_protocol_client_unknown_provider(monkeypatch):
    config = ProtocolConfig(provider="unknown", api_key="key")

    factories = {"deepseek": lambda cfg, deps: DummyClient(cfg)}
    monkeypatch.setitem(create_protocol_client.__globals__, "PROVIDER_FACTORIES", factories)

    with pytest.raises(ValueError, match="Неизвестный провайдер протокола"):
        create_protocol_client(config)


def test_create_protocol_service_returns_protocol_service(monkeypatch):
    client = DummyClient(config=None)

    class DummyProtocolService(ProtocolService):
        def __init__(self, client):
            super().__init__(client)

    monkeypatch.setitem(
        create_protocol_service.__globals__,
        "ProtocolService",
        DummyProtocolService,
    )

    service = create_protocol_service(client)
    assert isinstance(service, DummyProtocolService)

