"""Доменные модели транскрипции."""

from dataclasses import dataclass
from typing import List, Iterator, Union


@dataclass
class Segment:
    """Сегмент транскрипции с таймингами."""
    start: float
    end: float
    text: str
    
    def _format_time(self, seconds: float) -> str:
        """Форматирует время в секундах в читаемый формат MM:SS или HH:MM:SS.
        
        Args:
            seconds: Время в секундах
            
        Returns:
            Строка в формате MM:SS (если меньше часа) или HH:MM:SS (если больше часа)
        """
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    def to_line(self) -> str:
        """Форматирует сегмент в строку с таймингами в читаемом формате MM:SS или HH:MM:SS."""
        start_formatted = self._format_time(self.start)
        end_formatted = self._format_time(self.end)
        return f"[{start_formatted} - {end_formatted}] {self.text}"


@dataclass
class Transcript:
    """Транскрипция целиком."""
    segments: List[Segment]
    language: str
    model: str
    
    def to_text(self) -> str:
        """Возвращает текст транскрипции для записи в файл."""
        return '\n'.join(seg.to_line() for seg in self.segments)


