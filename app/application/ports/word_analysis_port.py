"""Порты для анализа слов."""

from abc import ABC, abstractmethod
from typing import Iterable


class ITextSource(ABC):
    """Источник текста для анализа."""

    @abstractmethod
    def read(self) -> str:
        """Возвращает текст для анализа."""
        raise NotImplementedError


class IStopwordsProvider(ABC):
    """Поставщик стоп-слов."""

    @abstractmethod
    def load(self) -> Iterable[str]:
        """Возвращает коллекцию стоп-слов."""
        raise NotImplementedError

