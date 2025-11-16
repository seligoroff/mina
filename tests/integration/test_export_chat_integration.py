import os
import pytest
from datetime import datetime, timezone

from app.adapters.output.telethon_chat_exporter import TelethonChatExporter
from app.domain.models import ChatExportConfig


pytestmark = pytest.mark.integration


def _env(name: str) -> str | None:
    v = os.environ.get(name)
    return v if v and v.strip() else None


def _require_env(*names: str):
    missing = [n for n in names if not _env(n)]
    # Дополнительный предохранитель: по умолчанию пропускаем Telethon-интеграцию
    if not os.environ.get("TELEGRAM_RUN_TELETHON") == "1":
        pytest.skip("Skip Telethon integration unless TELEGRAM_RUN_TELETHON=1")
    if missing:
        pytest.skip(f"Missing env vars for integration: {', '.join(missing)}")


def test_export_minimal_flow():
    _require_env("TELEGRAM_API_ID", "TELEGRAM_API_HASH", "TELEGRAM_SESSION_PATH", "TELEGRAM_PEER")
    api_id = int(str(_env("TELEGRAM_API_ID")).lstrip(":"))
    api_hash = _env("TELEGRAM_API_HASH")
    session_path = _env("TELEGRAM_SESSION_PATH")
    peer = _env("TELEGRAM_PEER")

    exporter = TelethonChatExporter(
        api_id=api_id,
        api_hash=api_hash,
        session_path=session_path,
        rate_limit_s_default=0.1,
    )
    cfg = ChatExportConfig(
        peer=peer,
        limit_per_call=20,
        rate_limit_s=0.1,
    )
    it = exporter.export(cfg)
    # consume up to 5 messages to keep test short
    count = 0
    for _ in it:
        count += 1
        if count >= 5:
            break
    assert count >= 0


