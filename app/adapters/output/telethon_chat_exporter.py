from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator, Callable, Iterable, Iterator, Optional

from app.application.ports import IChatExporter
from app.domain.exceptions import ChatExportError
from app.domain.models import ChatExportConfig, ChatMessage


class TelethonChatExporter(IChatExporter):
    """Реализация IChatExporter на базе Telethon.

    Инициализируется фабрикой клиента или параметрами (api_id, api_hash, session_path).
    Не выполняет запись в файлы, только отдаёт поток нормализованных сообщений.
    """

    def __init__(
        self,
        *,
        client_factory=None,
        api_id: Optional[int] = None,
        api_hash: Optional[str] = None,
        session_path: Optional[str] = None,
        rate_limit_s_default: float = 0.2,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ) -> None:
        self._client_factory = client_factory or self._default_client_factory
        self._api_id = api_id
        self._api_hash = api_hash
        self._session_path = session_path
        self._rate_limit_s_default = rate_limit_s_default
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor
        self._on_progress: Callable[[str], None] = lambda m: print(m, flush=True)

    def export(self, config: ChatExportConfig) -> Iterable[ChatMessage]:
        """Синхронный итератор по сообщениям, поверх внутреннего async-клиента."""
        # Оборачиваем асинхронный генератор в синхронный итератор
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            async_gen = self._export_async(config)
            async def consume() -> list[ChatMessage]:
                items: list[ChatMessage] = []
                async for item in async_gen:
                    items.append(item)
                return items
            for msg in loop.run_until_complete(consume()):
                yield msg
        except Exception as exc:  # noqa: BLE001
            raise ChatExportError(f"Ошибка экспорта чата: {exc}") from exc
        finally:
            try:
                loop.run_until_complete(asyncio.sleep(0))
            finally:
                loop.close()

    async def _export_async(self, config: ChatExportConfig) -> AsyncIterator[ChatMessage]:
        rate_limit_s = config.rate_limit_s if config.rate_limit_s is not None else self._rate_limit_s_default
        retries = 0
        delay = rate_limit_s
        from datetime import timezone as _tz

        def _ensure_aware_utc(dt: Optional[datetime]) -> Optional[datetime]:
            if dt is None:
                return None
            if dt.tzinfo is None:
                return dt.replace(tzinfo=_tz.utc)
            return dt.astimezone(_tz.utc)

        async with self._client() as client:
            # Разрешаем peer
            try:
                entity = await client.get_entity(config.peer)
            except Exception as exc:  # noqa: BLE001
                # Fallback: если передан числовой id канала/диалога без access_hash,
                # пытаемся найти сущность среди доступных диалогов.
                try:
                    target_id = int(str(config.peer).strip())
                except Exception:
                    target_id = None
                if target_id is None:
                    raise ChatExportError(f"Не удалось определить peer '{config.peer}': {exc}") from exc
                found = None
                try:
                    async for d in client.iter_dialogs(limit=2000):
                        ent = d.entity
                        ent_id = getattr(ent, "id", None)
                        if ent_id == target_id:
                            found = ent
                            break
                except Exception as scan_exc:  # noqa: BLE001
                    raise ChatExportError(f"Не удалось определить peer '{config.peer}': {exc}") from scan_exc
                if not found:
                    raise ChatExportError(f"Не удалось определить peer '{config.peer}': {exc}") from exc
                entity = found

            # Нормализуем границы по времени к aware UTC и итерируем в порядке возрастания по дате/ID
            date_from_utc: Optional[datetime] = _ensure_aware_utc(config.date_from)
            date_to_utc: Optional[datetime] = _ensure_aware_utc(config.date_to)
            # Стратегия итерации:
            # - если заданы обе границы, идём ВПЕРЁД от нижней (reverse=True) и выходим при достижении верхней — сохраняет глобальный возр. порядок
            # - если задана только верхняя граница, идём НАЗАД от верхней (reverse=False)
            # - иначе идём ВПЕРЁД от нижней границы (или с начала)
            go_backward = (date_to_utc is not None) and (date_from_utc is None)
            offset_date: Optional[datetime] = (date_to_utc if go_backward else date_from_utc)
            last_ts: Optional[int] = None

            while True:
                try:
                    msgs = []
                    async for m in client.iter_messages(
                        entity,
                        offset_date=offset_date,
                        reverse=not go_backward,
                        limit=config.limit_per_call,
                    ):
                        msgs.append(m)
                    if not msgs:
                        break
                    self._on_progress(f"получен батч: {len(msgs)} сообщений (offset={offset_date})")
                    # Для прямого направления (вперёд) — ранний выход при уходе за верхнюю границу
                    if not go_backward:
                        first_dt = _ensure_aware_utc(getattr(msgs[0], "date", None))
                        if date_to_utc and first_dt and first_dt > date_to_utc:
                            break
                    for m in msgs:
                        # Преобразование Telethon message -> ChatMessage
                        msg = self._to_chat_message(m)
                        # Фильтр по верхней границе даты (с нормализацией aware/naive)
                        md = _ensure_aware_utc(msg.date)
                        if date_to_utc and md and md > date_to_utc and not go_backward:
                            # Для прямого направления порядок возрастающий — дальше только позже
                            return
                        # Для обратного направления (движемся назад): если ушли ниже нижней границы — завершаем
                        if go_backward and date_from_utc and md and md < date_from_utc:
                            return
                        # монотонность: по времени/ID растет
                        if last_ts is not None and int(md.timestamp()) < last_ts:
                            # если нарушена монотонность — всё равно продолжаем, но фиксируем последнее значение
                            pass
                        last_ts = int(md.timestamp())
                        yield msg
                    # Подготовить следующий offset_date так, чтобы избежать повторов:
                    # - вперёд: после последнего сообщения батча
                    # - назад: перед первым сообщением батча
                    batch_first = _ensure_aware_utc(getattr(msgs[0], "date", None))
                    batch_last = _ensure_aware_utc(getattr(msgs[-1], "date", None))
                    from datetime import timedelta as _td
                    if not go_backward:
                        offset_date = (batch_last + _td(seconds=1)) if batch_last else offset_date
                    else:
                        offset_date = (batch_first - _td(seconds=1)) if batch_first else offset_date
                    # Rate limit
                    if rate_limit_s and rate_limit_s > 0:
                        await asyncio.sleep(rate_limit_s)
                    # сбросить счётчик ретраев после успешной пачки
                    retries = 0
                    delay = rate_limit_s or self._rate_limit_s_default
                except Exception as exc:  # noqa: BLE001
                    retries += 1
                    self._on_progress(f"временная ошибка: {exc} (попытка {retries}/{self._max_retries})")
                    if retries > self._max_retries:
                        raise ChatExportError(f"Сбой экспорта после {self._max_retries} попыток: {exc}") from exc
                    await asyncio.sleep(delay)
                    delay *= self._backoff_factor
                    continue

    @asynccontextmanager
    async def _client(self):
        client = self._client_factory(self._session_path, self._api_id, self._api_hash)
        try:
            await client.__aenter__()  # type: ignore[attr-defined]
            yield client
        finally:
            await client.__aexit__(None, None, None)  # type: ignore[attr-defined]

    @staticmethod
    def _default_client_factory(session_path: Optional[str], api_id: Optional[int], api_hash: Optional[str]):
        if not (session_path and api_id and api_hash):
            raise ChatExportError("Не заданы параметры Telethon клиента: session_path, api_id, api_hash")
        try:
            from telethon import TelegramClient  # type: ignore[import]
        except Exception as exc:  # noqa: BLE001
            raise ChatExportError(f"Не удалось импортировать Telethon: {exc}") from exc
        return TelegramClient(session_path, api_id, str(api_hash))

    @staticmethod
    def _to_chat_message(m) -> ChatMessage:
        # Минимальный маппинг необходимых полей
        msg_id = getattr(m, "id", None)
        date = getattr(m, "date", None)
        if msg_id is None or date is None:
            raise ChatExportError("Сообщение не содержит обязательных полей id/date")
        from_id = getattr(getattr(m, "from_id", None), "user_id", None)
        sender = getattr(m, "sender", None)
        from_name = None
        if sender is not None:
            first_name = getattr(sender, "first_name", None)
            last_name = getattr(sender, "last_name", None)
            if first_name or last_name:
                from_name = " ".join([p for p in [first_name, last_name] if p])
        text = getattr(m, "message", "") or getattr(m, "text", "") or ""
        reply_to = getattr(getattr(m, "reply_to", None), "reply_to_msg_id", None)
        # entities/attachments пока как пустые списки; заполнятся в расширении
        return ChatMessage(
            id=int(msg_id),
            date=date,
            from_id=from_id,
            from_name=from_name,
            text=text,
            reply_to=reply_to,
            entities=[],
            attachments=[],
        )


