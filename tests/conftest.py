"""Конфигурация и фикстуры для тестов (pytest)."""

import tempfile
import os
from pathlib import Path
import pytest
from unittest.mock import Mock


@pytest.fixture
def temp_dir():
    """Создает временную директорию для тестов."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_file(temp_dir):
    """Создает временный файл для тестов."""
    def _create_file(content="", suffix=".txt"):
        fd, path = tempfile.mkstemp(dir=temp_dir, suffix=suffix)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
        return path
    return _create_file


@pytest.fixture
def sample_dict_segments():
    """Пример сегментов в формате словаря (openai-whisper)."""
    return [
        {'start': 0.0, 'end': 1.5, 'text': 'Привет мир'},
        {'start': 1.5, 'end': 3.0, 'text': 'Как дела?'},
        {'start': 3.0, 'end': 5.0, 'text': 'Все отлично'}
    ]


@pytest.fixture
def sample_object_segments():
    """Пример сегментов в формате объекта (faster-whisper)."""
    segments = []
    for i, text in enumerate(['Первая фраза', 'Вторая фраза', 'Третья фраза']):
        seg = Mock()
        seg.start = i * 2.0
        seg.end = (i + 1) * 2.0
        seg.text = text
        segments.append(seg)
    return segments


@pytest.fixture
def sample_transcript_text():
    """Пример текста транскрипции с таймингами."""
    return """[0.00 - 1.50] Привет мир
[1.50 - 3.00] Как дела?
[3.00 - 5.00] Все отлично
"""


@pytest.fixture
def fixtures_dir():
    """Возвращает путь к директории с тестовыми данными."""
    return Path(__file__).parent / 'fixtures'


def pytest_configure(config):
    """Конфигурация pytest."""
    # Добавляем маркеры для категоризации тестов
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )

