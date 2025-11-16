from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Iterable

from app.application.ports import IMessageStore
from app.domain.models import ChatMessage


class FileMessageStore(IMessageStore):
    """Простой сторедж для записи сообщений в файл (JSONL или CSV)."""

    def write(self, messages: Iterable[ChatMessage], destination: str, format: str = "jsonl") -> None:
        fmt = (format or "jsonl").lower()
        if fmt == "jsonl":
            self._write_jsonl(messages, destination)
        elif fmt == "csv":
            self._write_csv(messages, destination)
        else:
            raise ValueError(f"Unsupported format: {format}")

    @staticmethod
    def _message_to_dict(message: ChatMessage) -> dict:
        if is_dataclass(message):
            data = asdict(message)
        else:
            # допускаем dict-совместимые объекты
            data = dict(message)  # type: ignore[arg-type]
        # приведение типов для JSONL
        if isinstance(data.get("date"), datetime):
            data["date"] = data["date"].isoformat()
        return data

    def _write_jsonl(self, messages: Iterable[ChatMessage], destination: str) -> None:
        with open(destination, "w", encoding="utf-8") as f:
            for msg in messages:
                data = self._message_to_dict(msg)
                f.write(json.dumps(data, ensure_ascii=False))
                f.write("\n")

    def _write_csv(self, messages: Iterable[ChatMessage], destination: str) -> None:
        # Поля CSV фиксируем, вложенные объекты сериализуем в JSON-строку
        fieldnames = [
            "id",
            "date",
            "from_id",
            "from_name",
            "text",
            "reply_to",
            "entities",
            "attachments",
        ]
        with open(destination, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            for msg in messages:
                data = self._message_to_dict(msg)
                row = {
                    "id": data.get("id"),
                    "date": data.get("date").isoformat() if data.get("date") else None,
                    "from_id": data.get("from_id"),
                    "from_name": data.get("from_name"),
                    "text": data.get("text"),
                    "reply_to": data.get("reply_to"),
                    "entities": json.dumps(data.get("entities", []), ensure_ascii=False),
                    "attachments": json.dumps(data.get("attachments", []), ensure_ascii=False),
                }
                writer.writerow(row)


