import pytest
from datetime import datetime, timezone

from app.adapters.output.telethon_chat_exporter import TelethonChatExporter
from app.domain.models import ChatExportConfig


class _FakeDialog:
    def __init__(self, entity):
        self.entity = entity


class _FakeEntity:
    def __init__(self, _id):
        self.id = _id


class _FakeMessage:
    def __init__(self, mid, dt):
        self.id = mid
        self.date = dt
        self.message = ""


class _ClientFailGetEntityThenIter:
    def __init__(self, entity_id, msgs):
        self._entity_id = entity_id
        self._msgs = msgs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get_entity(self, peer):
        raise RuntimeError("no access hash for numeric id")

    async def iter_dialogs(self, limit=2000):
        yield _FakeDialog(_FakeEntity(self._entity_id))

    async def iter_messages(self, entity, offset_date=None, reverse=True, limit=200):
        # Отдаём сообщения только один раз, далее — пусто (эмулируем пагинацию)
        batch = list(self._msgs)
        self._msgs = []
        for m in batch:
            yield m


def test_numeric_peer_fallback_resolves_dialog_and_exports():
    dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
    msgs = [_FakeMessage(1, dt)]
    client_factory = lambda s, a, h: _ClientFailGetEntityThenIter(2051202508, msgs)
    exporter = TelethonChatExporter(
        client_factory=client_factory, api_id=1, api_hash="x", session_path="/tmp/sess"
    )
    cfg = ChatExportConfig(peer="2051202508", limit_per_call=100)

    out = list(exporter.export(cfg))
    assert [m.id for m in out] == [1]


