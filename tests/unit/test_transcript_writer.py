"""Тесты для утилит."""

import pytest
from unittest.mock import Mock, mock_open, patch
from app.utils import write_transcript


@pytest.mark.unit
def test_write_transcript_with_dict_segments():
    """Тест записи транскрипции с сегментами в формате словаря (openai-whisper)."""
    segments = [
        {'start': 0.0, 'end': 1.5, 'text': 'Привет мир'},
        {'start': 1.5, 'end': 3.0, 'text': 'Как дела?'}
    ]
    output_path = '/tmp/test_transcript.txt'
    
    with patch('builtins.open', mock_open()) as mock_file:
        write_transcript(segments, output_path, verbose=False)
        
        # Проверяем, что файл был открыт для записи
        mock_file.assert_called_once_with(output_path, 'w', encoding='utf-8')
        
        # Проверяем, что были записаны строки
        handle = mock_file()
        assert handle.write.call_count == 2
        
        # Проверяем формат строк
        calls = handle.write.call_args_list
        assert '[0.00 - 1.50]' in calls[0][0][0]
        assert 'Привет мир' in calls[0][0][0]
        assert '[1.50 - 3.00]' in calls[1][0][0]
        assert 'Как дела?' in calls[1][0][0]


@pytest.mark.unit
def test_write_transcript_with_object_segments():
    """Тест записи транскрипции с сегментами в формате объекта (faster-whisper)."""
    # Создаем мок-объекты с атрибутами
    segment1 = Mock()
    segment1.start = 0.0
    segment1.end = 2.5
    segment1.text = '  Первая фраза  '
    
    segment2 = Mock()
    segment2.start = 2.5
    segment2.end = 4.0
    segment2.text = 'Вторая фраза'
    
    segments = [segment1, segment2]
    output_path = '/tmp/test_transcript.txt'
    
    with patch('builtins.open', mock_open()) as mock_file:
        write_transcript(segments, output_path, verbose=False)
        
        # Проверяем, что файл был открыт
        mock_file.assert_called_once_with(output_path, 'w', encoding='utf-8')
        
        # Проверяем формат - пробелы должны быть удалены
        handle = mock_file()
        calls = handle.write.call_args_list
        assert '[0.00 - 2.50]' in calls[0][0][0]
        assert 'Первая фраза' in calls[0][0][0]  # Без пробелов
        assert '  Первая фраза  ' not in calls[0][0][0]  # Пробелы удалены


@pytest.mark.unit
def test_write_transcript_verbose_mode():
    """Тест записи транскрипции с verbose=True (вывод в консоль)."""
    segments = [
        {'start': 0.0, 'end': 1.0, 'text': 'Тест'}
    ]
    output_path = '/tmp/test_transcript.txt'
    
    with patch('builtins.open', mock_open()), \
         patch('builtins.print') as mock_print:
        
        write_transcript(segments, output_path, verbose=True)
        
        # Проверяем, что print был вызван
        mock_print.assert_called_once()
        # Проверяем, что в print передана правильная строка
        printed_line = mock_print.call_args[0][0]
        assert '[0.00 - 1.00]' in printed_line
        assert 'Тест' in printed_line


@pytest.mark.unit
def test_write_transcript_empty_segments():
    """Тест записи пустой транскрипции."""
    segments = []
    output_path = '/tmp/test_transcript.txt'
    
    with patch('builtins.open', mock_open()) as mock_file:
        write_transcript(segments, output_path, verbose=False)
        
        # Файл должен быть создан, но пустой
        handle = mock_file()
        handle.write.assert_not_called()


@pytest.mark.unit
def test_write_transcript_with_generator():
    """Тест записи транскрипции с генератором (faster-whisper стиль)."""
    def segment_generator():
        for i in range(3):
            seg = Mock()
            seg.start = i * 2.0
            seg.end = (i + 1) * 2.0
            seg.text = f'Фраза {i}'
            yield seg
    
    output_path = '/tmp/test_transcript.txt'
    
    with patch('builtins.open', mock_open()) as mock_file:
        write_transcript(segment_generator(), output_path, verbose=False)
        
        # Проверяем, что все сегменты были записаны
        handle = mock_file()
        assert handle.write.call_count == 3

