from __future__ import annotations

from typing import Any, Dict, Optional

from app.adapters.output.telethon_chat_exporter import TelethonChatExporter
from app.application.ports import IChatExporter


def create_telethon_chat_exporter(dependencies: Optional[Dict[str, Any]] = None) -> IChatExporter:
    """Фабрика экспортера чатов на Telethon.

    Ожидаемые зависимости:
    - api_id: int
    - api_hash: str
    - session_path: str
    - rate_limit_s: float (опционально)
    """
    deps = dependencies or {}
    api_id = deps.get("api_id")
    api_hash = deps.get("api_hash")
    session_path = deps.get("session_path")
    rate_limit_s = deps.get("rate_limit_s", 0.2)

    return TelethonChatExporter(
        api_id=api_id,
        api_hash=api_hash,
        session_path=session_path,
        rate_limit_s_default=rate_limit_s,
    )


