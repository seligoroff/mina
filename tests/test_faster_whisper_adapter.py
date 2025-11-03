"""Тесты для FasterWhisperAdapter."""

import pytest
from unittest.mock import Mock, MagicMock
from app.adapters.output.whisper import FasterWhisperAdapter
from app.domain.models.transcript import Segment


@pytest.mark.unit
class TestFasterWhisperAdapter:
    """Тесты для FasterWhisperAdapter."""
    
    def test_init(self):
        """Тест инициализации адаптера."""
        faster_whisper_model_class = Mock()
        adapter = FasterWhisperAdapter(faster_whisper_model_class)
        
        assert adapter._faster_whisper_model_class is faster_whisper_model_class
        assert adapter._compute_type == 'int8'  # Значение по умолчанию
    
    def test_init_with_custom_compute_type(self):
        """Тест инициализации адаптера с кастомным compute_type."""
        faster_whisper_model_class = Mock()
        adapter = FasterWhisperAdapter(faster_whisper_model_class, compute_type="float16")
        
        assert adapter._faster_whisper_model_class is faster_whisper_model_class
        assert adapter._compute_type == "float16"
    
    def test_load_model_with_default_compute_type(self):
        """Тест загрузки модели с compute_type по умолчанию."""
        faster_whisper_model_class = Mock()
        mock_model = Mock()
        faster_whisper_model_class.return_value = mock_model
        
        adapter = FasterWhisperAdapter(faster_whisper_model_class)
        
        result = adapter.load_model("base")
        
        # Проверяем, что модель создана с compute_type='int8' по умолчанию
        faster_whisper_model_class.assert_called_once_with("base", compute_type="int8")
        assert result is mock_model
    
    def test_load_model_with_custom_compute_type(self):
        """Тест загрузки модели с кастомным compute_type в конструкторе."""
        faster_whisper_model_class = Mock()
        mock_model = Mock()
        faster_whisper_model_class.return_value = mock_model
        
        # Создаем адаптер с compute_type="float16" в конструкторе
        adapter = FasterWhisperAdapter(faster_whisper_model_class, compute_type="float16")
        
        result = adapter.load_model("small")
        
        # Проверяем, что compute_type из конструктора использован
        faster_whisper_model_class.assert_called_once_with("small", compute_type="float16")
        assert result is mock_model
        assert adapter._compute_type == "float16"
    
    def test_load_model_with_float32_compute_type(self):
        """Тест загрузки модели с compute_type=float32 в конструкторе."""
        faster_whisper_model_class = Mock()
        mock_model = Mock()
        faster_whisper_model_class.return_value = mock_model
        
        # Создаем адаптер с compute_type="float32" в конструкторе
        adapter = FasterWhisperAdapter(faster_whisper_model_class, compute_type="float32")
        
        result = adapter.load_model("medium")
        
        # Проверяем, что compute_type из конструктора использован
        faster_whisper_model_class.assert_called_once_with("medium", compute_type="float32")
        assert result is mock_model
        assert adapter._compute_type == "float32"
    
    def test_transcribe_converts_objects_to_segments(self):
        """Тест конвертации объектов faster-whisper в доменные модели Segment."""
        faster_whisper_model_class = Mock()
        adapter = FasterWhisperAdapter(faster_whisper_model_class)
        
        # Создаем мок модель
        mock_model = Mock()
        
        # Создаем мок сегменты с атрибутами (как в faster-whisper)
        segment1 = Mock()
        segment1.start = 0.0
        segment1.end = 2.5
        segment1.text = "  Первая фраза  "
        
        segment2 = Mock()
        segment2.start = 2.5
        segment2.end = 4.0
        segment2.text = "Вторая фраза"
        
        segment3 = Mock()
        segment3.start = 4.0
        segment3.end = 6.0
        segment3.text = "Третья фраза"
        
        # faster-whisper возвращает (segments, info) tuple
        mock_model.transcribe.return_value = ([segment1, segment2, segment3], {})
        
        # Выполняем транскрипцию
        segments = list(adapter.transcribe(
            model=mock_model,
            audio_path="/path/to/audio.mp3",
            language="ru"
        ))
        
        # Проверяем, что model.transcribe был вызван правильно
        mock_model.transcribe.assert_called_once_with(
            "/path/to/audio.mp3",
            beam_size=5,  # default
            language="ru"
        )
        
        # Проверяем количество сегментов
        assert len(segments) == 3
        
        # Проверяем первый сегмент
        assert isinstance(segments[0], Segment)
        assert segments[0].start == 0.0
        assert segments[0].end == 2.5
        assert segments[0].text == "Первая фраза"  # пробелы удалены
        
        # Проверяем второй сегмент
        assert segments[1].start == 2.5
        assert segments[1].end == 4.0
        assert segments[1].text == "Вторая фраза"
        
        # Проверяем третий сегмент
        assert segments[2].start == 4.0
        assert segments[2].end == 6.0
        assert segments[2].text == "Третья фраза"
    
    def test_transcribe_with_custom_beam_size(self):
        """Тест транскрипции с кастомным beam_size."""
        faster_whisper_model_class = Mock()
        adapter = FasterWhisperAdapter(faster_whisper_model_class)
        
        mock_model = Mock()
        mock_model.transcribe.return_value = ([], {})
        
        list(adapter.transcribe(
            model=mock_model,
            audio_path="/path/to/audio.mp3",
            language="en",
            beam_size=10
        ))
        
        # Проверяем, что beam_size=10 был передан
        mock_model.transcribe.assert_called_once_with(
            "/path/to/audio.mp3",
            beam_size=10,
            language="en"
        )
    
    def test_transcribe_with_default_beam_size(self):
        """Тест, что beam_size=5 используется по умолчанию."""
        faster_whisper_model_class = Mock()
        adapter = FasterWhisperAdapter(faster_whisper_model_class)
        
        mock_model = Mock()
        mock_model.transcribe.return_value = ([], {})
        
        list(adapter.transcribe(
            model=mock_model,
            audio_path="/path/to/audio.mp3",
            language="ru"
        ))
        
        # Проверяем, что beam_size=5 используется по умолчанию
        mock_model.transcribe.assert_called_once_with(
            "/path/to/audio.mp3",
            beam_size=5,
            language="ru"
        )
    
    def test_transcribe_empty_segments(self):
        """Тест транскрипции с пустыми сегментами."""
        faster_whisper_model_class = Mock()
        adapter = FasterWhisperAdapter(faster_whisper_model_class)
        
        mock_model = Mock()
        mock_model.transcribe.return_value = ([], {})
        
        segments = list(adapter.transcribe(
            model=mock_model,
            audio_path="/path/to/audio.mp3",
            language="ru"
        ))
        
        assert len(segments) == 0
    
    def test_transcribe_strips_text(self):
        """Тест, что текст сегментов обрезается от пробелов."""
        faster_whisper_model_class = Mock()
        adapter = FasterWhisperAdapter(faster_whisper_model_class)
        
        mock_model = Mock()
        segment = Mock()
        segment.start = 0.0
        segment.end = 1.0
        segment.text = "  текст с пробелами  "
        
        mock_model.transcribe.return_value = ([segment], {})
        
        segments = list(adapter.transcribe(
            model=mock_model,
            audio_path="/path/to/audio.mp3",
            language="ru"
        ))
        
        assert segments[0].text == "текст с пробелами"  # пробелы удалены
    
    def test_transcribe_ignores_info_tuple(self):
        """Тест, что второй элемент tuple (info) игнорируется."""
        faster_whisper_model_class = Mock()
        adapter = FasterWhisperAdapter(faster_whisper_model_class)
        
        mock_model = Mock()
        segment = Mock()
        segment.start = 0.0
        segment.end = 1.0
        segment.text = "Тест"
        
        info = {"language": "ru", "language_probability": 0.9}
        mock_model.transcribe.return_value = ([segment], info)
        
        segments = list(adapter.transcribe(
            model=mock_model,
            audio_path="/path/to/audio.mp3",
            language="ru"
        ))
        
        # Проверяем, что сегмент обработан, несмотря на наличие info
        assert len(segments) == 1
        assert segments[0].text == "Тест"

