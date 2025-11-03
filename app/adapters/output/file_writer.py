"""Адаптер для записи транскрипции в файл."""

from typing import Optional
from app.application.ports.output_port import ITranscriptSegmentWriter
from app.domain.models.transcript import Segment


class FileOutputWriter(ITranscriptSegmentWriter):
    """Адаптер для записи транскрипции в файл с выводом в консоль.
    
    Оборачивает файловые операции и вывод в консоль, адаптируя их
    к интерфейсу ITranscriptSegmentWriter.
    """
    
    def __init__(self, output_path: str, verbose: bool = False):
        """
        Args:
            output_path: Путь к выходному файлу
            verbose: Если True, выводит сегменты на экран в реальном времени
        """
        self._output_path = output_path
        self._verbose = verbose
        self._file = open(output_path, "w", encoding="utf-8")
    
    def write_segment(self, segment: Segment) -> None:
        """Записывает сегмент транскрипции в файл и на экран (если verbose=True).
        
        Args:
            segment: Сегмент транскрипции для записи
        """
        line = segment.to_line() + "\n"
        
        # Записываем в файл
        self._file.write(line)
        # Сбрасываем буфер после каждой записи, чтобы данные не терялись
        self._file.flush()
        
        # Выводим на экран в реальном времени, если verbose=True
        if self._verbose:
            print(line.strip())
    
    def close(self) -> None:
        """Завершает запись и закрывает файл."""
        self._file.close()

