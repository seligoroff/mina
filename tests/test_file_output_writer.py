"""Тесты для FileOutputWriter."""

import pytest
from app.adapters.output.file_writer import FileOutputWriter
from app.domain.models.transcript import Segment
from app.application.ports.output_port import ITranscriptSegmentWriter


@pytest.mark.unit
class TestFileOutputWriter:
    """Тесты для FileOutputWriter."""
    
    def test_file_output_writer_init_creates_file(self, tmp_path):
        """Тест создания файла при инициализации."""
        output_file = tmp_path / "test_output.txt"
        writer = FileOutputWriter(str(output_file), verbose=False)
        
        assert output_file.exists()
        writer.close()
    
    def test_file_output_writer_init_opens_file_in_write_mode(self, tmp_path):
        """Тест открытия файла в режиме записи."""
        output_file = tmp_path / "test_output.txt"
        writer = FileOutputWriter(str(output_file), verbose=False)
        
        assert writer._file.mode == 'w'
        assert writer._file.encoding == 'utf-8'
        writer.close()
    
    def test_file_output_writer_write_segment(self, tmp_path):
        """Тест записи сегмента в файл."""
        output_file = tmp_path / "test_output.txt"
        writer = FileOutputWriter(str(output_file), verbose=False)
        
        segment = Segment(start=0.0, end=1.5, text="Тест")
        writer.write_segment(segment)
        writer.close()
        
        # Проверяем запись в файл (новый формат MM:SS)
        content = output_file.read_text(encoding='utf-8')
        assert "[0:00 - 0:01] Тест\n" in content
    
    def test_file_output_writer_write_segment_verbose(self, tmp_path, capsys):
        """Тест вывода сегмента в консоль при verbose=True."""
        output_file = tmp_path / "test_output.txt"
        writer = FileOutputWriter(str(output_file), verbose=True)
        
        segment = Segment(start=0.0, end=1.5, text="Тест")
        writer.write_segment(segment)
        writer.close()
        
        # Проверяем вывод в консоль (verbose=True) (новый формат MM:SS)
        captured = capsys.readouterr()
        assert "[0:00 - 0:01] Тест" in captured.out
    
    def test_file_output_writer_write_segment_no_verbose(self, tmp_path, capsys):
        """Тест отсутствия вывода в консоль при verbose=False."""
        output_file = tmp_path / "test_output.txt"
        writer = FileOutputWriter(str(output_file), verbose=False)
        
        segment = Segment(start=0.0, end=1.5, text="Тест")
        writer.write_segment(segment)
        writer.close()
        
        # Проверяем отсутствие вывода в консоль
        captured = capsys.readouterr()
        assert captured.out == ""
    
    def test_file_output_writer_write_multiple_segments(self, tmp_path):
        """Тест записи нескольких сегментов."""
        output_file = tmp_path / "test_output.txt"
        writer = FileOutputWriter(str(output_file), verbose=False)
        
        segments = [
            Segment(start=0.0, end=1.5, text="Первый"),
            Segment(start=1.5, end=3.0, text="Второй"),
            Segment(start=3.0, end=4.5, text="Третий"),
        ]
        
        for segment in segments:
            writer.write_segment(segment)
        
        writer.close()
        
        # Проверяем запись всех сегментов (новый формат MM:SS)
        content = output_file.read_text(encoding='utf-8')
        assert "[0:00 - 0:01] Первый\n" in content
        assert "[0:01 - 0:03] Второй\n" in content
        assert "[0:03 - 0:04] Третий\n" in content
    
    def test_file_output_writer_close_closes_file(self, tmp_path):
        """Тест закрытия файла."""
        output_file = tmp_path / "test_output.txt"
        writer = FileOutputWriter(str(output_file), verbose=False)
        
        assert not writer._file.closed
        writer.close()
        assert writer._file.closed
    
    def test_file_output_writer_close_can_be_called_multiple_times(self, tmp_path):
        """Тест безопасного множественного вызова close()."""
        output_file = tmp_path / "test_output.txt"
        writer = FileOutputWriter(str(output_file), verbose=False)
        
        writer.close()
        assert writer._file.closed
        
        # Повторный вызов не должен вызвать ошибку
        writer.close()
        assert writer._file.closed
    
    def test_file_output_writer_implements_ioutput_writer(self, tmp_path):
        """Тест реализации интерфейса ITranscriptSegmentWriter."""
        output_file = tmp_path / "dummy.txt"
        writer = FileOutputWriter(str(output_file), verbose=False)
        
        assert isinstance(writer, ITranscriptSegmentWriter)
        writer.close()
    
    def test_file_output_writer_segment_formatting(self, tmp_path):
        """Тест форматирования сегмента при записи."""
        output_file = tmp_path / "test_output.txt"
        writer = FileOutputWriter(str(output_file), verbose=False)
        
        segment = Segment(start=123.456, end=789.012, text="Текст сегмента")
        writer.write_segment(segment)
        writer.close()
        
        # Проверяем форматирование (новый формат MM:SS)
        # 123 секунды = 2 минуты 3 секунды, 789 секунд = 13 минут 9 секунд
        content = output_file.read_text(encoding='utf-8')
        assert "[2:03 - 13:09] Текст сегмента\n" in content
    
    def test_file_output_writer_strips_output_in_console(self, tmp_path, capsys):
        """Тест обрезки строки при выводе в консоль (без \n)."""
        output_file = tmp_path / "test_output.txt"
        writer = FileOutputWriter(str(output_file), verbose=True)
        
        segment = Segment(start=0.0, end=1.5, text="Тест")
        writer.write_segment(segment)
        writer.close()
        
        # Проверяем, что в консоль выводится без \n (новый формат MM:SS)
        captured = capsys.readouterr()
        assert "\n" not in captured.out or captured.out.strip() == "[0:00 - 0:01] Тест"
    
    def test_file_output_writer_empty_segment(self, tmp_path):
        """Тест записи сегмента с пустым текстом."""
        output_file = tmp_path / "test_output.txt"
        writer = FileOutputWriter(str(output_file), verbose=False)
        
        segment = Segment(start=0.0, end=1.0, text="")
        writer.write_segment(segment)
        writer.close()
        
        content = output_file.read_text(encoding='utf-8')
        assert "[0:00 - 0:01] \n" in content


