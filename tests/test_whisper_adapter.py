"""Тесты для WhisperAdapter."""

import pytest
from unittest.mock import Mock, MagicMock
from app.adapters.output.whisper import WhisperAdapter
from app.domain.models.transcript import Segment


@pytest.mark.unit
class TestWhisperAdapter:
    """Тесты для WhisperAdapter."""
    
    def test_init(self):
        """Тест инициализации адаптера."""
        whisper_module = Mock()
        adapter = WhisperAdapter(whisper_module)
        
        assert adapter._whisper is whisper_module
    
    def test_load_model(self):
        """Тест загрузки модели через Whisper."""
        whisper_module = Mock()
        mock_model = Mock()
        whisper_module.load_model.return_value = mock_model
        
        adapter = WhisperAdapter(whisper_module)
        model_name = "base"
        
        result = adapter.load_model(model_name)
        
        # Проверяем, что load_model был вызван с правильными параметрами
        whisper_module.load_model.assert_called_once_with(model_name)
        assert result is mock_model
    
    def test_load_model_ignores_kwargs(self):
        """Тест, что дополнительные kwargs игнорируются для Whisper."""
        whisper_module = Mock()
        mock_model = Mock()
        whisper_module.load_model.return_value = mock_model
        
        adapter = WhisperAdapter(whisper_module)
        
        # Передаем compute_type, который используется только в faster-whisper
        result = adapter.load_model("small", compute_type="int8")
        
        # Whisper должен получить только model_name
        whisper_module.load_model.assert_called_once_with("small")
        assert result is mock_model
    
    def test_transcribe_converts_dict_to_segments(self):
        """Тест конвертации словарей Whisper в доменные модели Segment."""
        whisper_module = Mock()
        adapter = WhisperAdapter(whisper_module)
        
        # Создаем мок модель
        mock_model = Mock()
        whisper_result = {
            'segments': [
                {'start': 0.0, 'end': 1.5, 'text': '  Привет мир  '},
                {'start': 1.5, 'end': 3.0, 'text': 'Как дела?'},
                {'start': 3.0, 'end': 5.0, 'text': 'Все отлично'}
            ]
        }
        mock_model.transcribe.return_value = whisper_result
        
        # Выполняем транскрипцию
        segments = list(adapter.transcribe(
            model=mock_model,
            audio_path="/path/to/audio.mp3",
            language="ru"
        ))
        
        # Проверяем, что model.transcribe был вызван правильно
        mock_model.transcribe.assert_called_once_with(
            "/path/to/audio.mp3",
            language="ru",
            verbose=True  # default
        )
        
        # Проверяем количество сегментов
        assert len(segments) == 3
        
        # Проверяем первый сегмент
        assert isinstance(segments[0], Segment)
        assert segments[0].start == 0.0
        assert segments[0].end == 1.5
        assert segments[0].text == "Привет мир"  # пробелы удалены
        
        # Проверяем второй сегмент
        assert segments[1].start == 1.5
        assert segments[1].end == 3.0
        assert segments[1].text == "Как дела?"
        
        # Проверяем третий сегмент
        assert segments[2].start == 3.0
        assert segments[2].end == 5.0
        assert segments[2].text == "Все отлично"
    
    def test_transcribe_with_verbose_false(self):
        """Тест транскрипции с verbose=False."""
        whisper_module = Mock()
        adapter = WhisperAdapter(whisper_module)
        
        mock_model = Mock()
        whisper_result = {'segments': [{'start': 0.0, 'end': 1.0, 'text': 'Тест'}]}
        mock_model.transcribe.return_value = whisper_result
        
        list(adapter.transcribe(
            model=mock_model,
            audio_path="/path/to/audio.mp3",
            language="en",
            verbose=False
        ))
        
        # Проверяем, что verbose=False был передан
        mock_model.transcribe.assert_called_once_with(
            "/path/to/audio.mp3",
            language="en",
            verbose=False
        )
    
    def test_transcribe_with_default_verbose(self):
        """Тест, что verbose=True по умолчанию."""
        whisper_module = Mock()
        adapter = WhisperAdapter(whisper_module)
        
        mock_model = Mock()
        whisper_result = {'segments': []}
        mock_model.transcribe.return_value = whisper_result
        
        list(adapter.transcribe(
            model=mock_model,
            audio_path="/path/to/audio.mp3",
            language="ru"
        ))
        
        # Проверяем, что verbose=True используется по умолчанию
        mock_model.transcribe.assert_called_once_with(
            "/path/to/audio.mp3",
            language="ru",
            verbose=True
        )
    
    def test_transcribe_empty_segments(self):
        """Тест транскрипции с пустыми сегментами."""
        whisper_module = Mock()
        adapter = WhisperAdapter(whisper_module)
        
        mock_model = Mock()
        whisper_result = {'segments': []}
        mock_model.transcribe.return_value = whisper_result
        
        segments = list(adapter.transcribe(
            model=mock_model,
            audio_path="/path/to/audio.mp3",
            language="ru"
        ))
        
        assert len(segments) == 0
    
    def test_transcribe_strips_text(self):
        """Тест, что текст сегментов обрезается от пробелов."""
        whisper_module = Mock()
        adapter = WhisperAdapter(whisper_module)
        
        mock_model = Mock()
        whisper_result = {
            'segments': [
                {'start': 0.0, 'end': 1.0, 'text': '  текст с пробелами  '}
            ]
        }
        mock_model.transcribe.return_value = whisper_result
        
        segments = list(adapter.transcribe(
            model=mock_model,
            audio_path="/path/to/audio.mp3",
            language="ru"
        ))
        
        assert segments[0].text == "текст с пробелами"  # пробелы удалены

