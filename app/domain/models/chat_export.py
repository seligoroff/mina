from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal, Optional


@dataclass(frozen=True)
class ChatExportConfig:
    """Параметры экспорта чата/канала. Доменная модель без зависимостей от Telethon."""

    peer: str
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit_per_call: int = 200
    include_media: bool = False
    output_format: Literal["jsonl", "csv"] = "jsonl"
    rate_limit_s: float = 0.2


@dataclass(frozen=True)
class ChatMessage:
    """Нормализованное сообщение чата, содержащее только примитивные типы."""

    id: int
    date: datetime
    from_id: Optional[int] = None
    from_name: Optional[str] = None
    text: str = ""
    reply_to: Optional[int] = None
    entities: List[Dict[str, object]] = field(default_factory=list)
    attachments: List[Dict[str, object]] = field(default_factory=list)


