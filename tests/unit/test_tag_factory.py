"""Тесты для tag_factory."""

from unittest.mock import Mock

from app.factories.tag_factory import create_word_analysis_service


def test_create_word_analysis_service_with_dependencies():
    morph = Mock()
    service = create_word_analysis_service(dependencies={"morph": morph})
    assert service._morph is morph


def test_create_word_analysis_service_creates_default_analyzer(monkeypatch):
    created = {}

    class FakeMorph:
        pass

    def fake_morph_analyzer(lang):
        created["lang"] = lang
        return FakeMorph()

    monkeypatch.setattr(
        create_word_analysis_service.__globals__["pymorphy3"],
        "MorphAnalyzer",
        fake_morph_analyzer,
    )

    service = create_word_analysis_service()
    assert isinstance(service._morph, FakeMorph)
    assert created["lang"] == "ru"


