import asyncio
from datetime import datetime, timedelta, timezone
import pytest

from app.adapters.output.telethon_chat_exporter import TelethonChatExporter
from app.domain.models import ChatExportConfig
from app.domain.exceptions import ChatExportError


class _DummyFromId:
    def __init__(self, user_id):
        self.user_id = user_id


class _DummyReplyTo:
    def __init__(self, reply_to_msg_id):
        self.reply_to_msg_id = reply_to_msg_id


class _DummySender:
    def __init__(self, first_name=None, last_name=None):
        self.first_name = first_name
        self.last_name = last_name


class _DummyMessage:
    def __init__(self, mid, date, text, from_id=None, sender=None, reply_to=None):
        self.id = mid
        self.date = date
        self.message = text
        self.from_id = from_id
        self.sender = sender
        self.reply_to = reply_to


def test_to_chat_message_mapping():
    dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    msg = _DummyMessage(
        mid=1,
        date=dt,
        text="hello",
        from_id=_DummyFromId(42),
        sender=_DummySender(first_name="Ivan", last_name="Petrov"),
        reply_to=_DummyReplyTo(5),
    )
    cm = TelethonChatExporter._to_chat_message(msg)
    assert cm.id == 1
    assert cm.date == dt
    assert cm.from_id == 42
    assert cm.from_name == "Ivan Petrov"
    assert cm.text == "hello"
    assert cm.reply_to == 5
    assert cm.entities == []
    assert cm.attachments == []


class _FakeClient:
    def __init__(self, pages, fail_first=False):
        # pages: list[list[_DummyMessage]]
        self._pages = list(pages)
        self._fail_first = fail_first
        self._entered = False
        self._calls = 0

    async def __aenter__(self):
        self._entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._entered = False
        return False

    async def get_entity(self, peer):
        return object()

    async def iter_messages(self, entity, offset_date=None, reverse=True, limit=200):
        # simulate failure on first call for retry test
        self._calls += 1
        if self._fail_first and self._calls == 1:
            raise RuntimeError("temporary error")
        # yield next page or nothing
        if not self._pages:
            if False:
                yield  # pragma: no cover
            return
        page = self._pages.pop(0)
        for m in page:
            yield m


def test_pagination_and_limits_with_retries():
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    pages = [
        [_DummyMessage(1, base + timedelta(minutes=1), "a")],
        [_DummyMessage(2, base + timedelta(minutes=2), "b"),
         _DummyMessage(3, base + timedelta(minutes=3), "c")],
    ]

    def client_factory(session_path, api_id, api_hash):
        return _FakeClient(pages=[list(p) for p in pages], fail_first=True)

    exporter = TelethonChatExporter(
        client_factory=client_factory,
        api_id=1,
        api_hash="x",
        session_path="/tmp/sess",
        rate_limit_s_default=0.0,
        max_retries=2,
        backoff_factor=1.0,
    )

    cfg = ChatExportConfig(peer="any", limit_per_call=2, rate_limit_s=0.0)
    out = list(exporter.export(cfg))
    assert [m.id for m in out] == [1, 2, 3]
    assert out[0].date <= out[-1].date


def test_error_when_missing_required_fields():
    bad = _DummyMessage(mid=None, date=None, text="")
    with pytest.raises(ChatExportError):
        TelethonChatExporter._to_chat_message(bad)


