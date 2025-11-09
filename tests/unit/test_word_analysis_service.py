"""Тесты для WordAnalysisService."""

from typing import Iterable, List

import pytest

from app.application.services.word_analysis import WordAnalysisService
from app.domain.models.word_analysis import WordAnalysisConfig


class FakeParsed:
    def __init__(self, word: str, normal_form: str, pos: str = "NOUN", grammemes=None):
        self.word = word
        self.normal_form = normal_form
        self.tag = FakeTag(pos, grammemes or set())


class FakeTag:
    def __init__(self, pos: str, grammemes):
        self.POS = pos
        self.grammemes = set(grammemes)


class FakeMorph:
    def __init__(self, mapping: dict):
        self.mapping = mapping

    def parse(self, word: str) -> List[FakeParsed]:
        entry = self.mapping.get(word, (word, "NOUN", set()))
        normal, pos, grammemes = entry
        return [FakeParsed(word, normal, pos, grammemes)]


@pytest.fixture
def morph():
    mapping = {
        "работы": ("работа", "NOUN", {"Sing"}),
        "делать": ("делать", "VERB", {"INFN"}),
        "иванов": ("иванов", "NOUN", {"Name"}),
    }
    return FakeMorph(mapping)


@pytest.fixture
def service(morph):
    return WordAnalysisService(morph_analyzer=morph)


def test_extract_text_removes_timestamps(service):
    lines = [
        "[00:00 - 00:05] Привет всем",
        "Без таймкода строка",
        "[00:05 - 00:10] Ещё текст",
    ]
    result = service.extract_text(lines)
    assert "привет всем" in result
    assert "без таймкода строка" in result
    assert "ещё текст" in result


def test_analyze_basic_flow(service):
    lines = ["Привет привет друзья"]
    config = WordAnalysisConfig(limit=10)
    result = service.analyze(lines=lines, stopwords=[], config=config)
    assert result.items == [("привет", 2), ("друзья", 1)]


def test_analyze_with_lemmatization_and_filters(service):
    lines = ["Работы делать Иванов"]
    config = WordAnalysisConfig(limit=10, lemmatize=True, exclude_names=True)
    result = service.analyze(lines=lines, stopwords=[], config=config)
    # "иванов" исключён по Name
    assert ("работа", 1) in result.items
    assert all(word != "иванов" for word, _ in result.items)


def test_analyze_applies_stopwords(service):
    lines = ["раз два три раз"]
    config = WordAnalysisConfig(limit=10)
    result = service.analyze(lines=lines, stopwords=["раз"], config=config)
    assert ("три", 1) in result.items
    assert all(word != "раз" for word, _ in result.items)


def test_analyze_raises_when_no_words(service):
    with pytest.raises(ValueError):
        service.analyze(lines=[".."], stopwords=[], config=WordAnalysisConfig())


def test_analyze_raises_after_filters(service):
    lines = ["Иванов"]
    config = WordAnalysisConfig(limit=10, lemmatize=True, exclude_names=True)
    with pytest.raises(ValueError):
        service.analyze(lines=lines, stopwords=[], config=config)

