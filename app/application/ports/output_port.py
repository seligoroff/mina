"""Порт (интерфейс) для записи результатов транскрипции."""

from abc import ABC, abstractmethod
from app.domain.models.transcript import Segment


class ITranscriptSegmentWriter(ABC):
    """Порт (интерфейс) для записи сегментов транскрипции.
    
    Этот интерфейс абстрагирует различные способы вывода сегментов транскрипции
    (файл, консоль, БД, API и т.д.), что позволяет:
    - Легко заменять способы вывода без изменения бизнес-логики
    - Мокировать в тестах
    - Следовать принципу инверсии зависимостей (Dependency Inversion Principle)
    """
    
    @abstractmethod
    def write_segment(self, segment: Segment) -> None:
        """Записывает сегмент транскрипции.
        
        Args:
            segment: Сегмент транскрипции для записи
        """
        ...
    
    @abstractmethod
    def close(self) -> None:
        """Завершает запись и освобождает ресурсы.
        
        Этот метод должен вызываться после завершения записи всех сегментов.
        """
        ...


