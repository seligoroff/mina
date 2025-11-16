"""Доменные модели для анализа слов."""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class WordAnalysisConfig:
    """Конфигурация анализа слов."""

    lemmatize: bool = False
    exclude_names: bool = False
    limit: int = 50
    stopwords: Optional[Sequence[str]] = None


@dataclass(frozen=True)
class WordFrequencyResult:
    """Результат анализа слов."""

    items: List[Tuple[str, int]]

    def top(self, limit: Optional[int] = None) -> List[Tuple[str, int]]:
        """Возвращает первые N элементов (по умолчанию весь список)."""
        if limit is None or limit >= len(self.items):
            return list(self.items)
        return self.items[:limit]

    def to_text(self) -> str:
        """Форматирует результат в текстовый вид."""
        return "\n".join(f"{word}: {count}" for word, count in self.items)





