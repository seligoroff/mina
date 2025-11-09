import pytest

from app.domain.models.word_analysis import WordFrequencyResult


@pytest.mark.unit
def test_word_frequency_result_top_with_limit():
    result = WordFrequencyResult(items=[("alpha", 5), ("beta", 3), ("gamma", 1)])

    top_two = result.top(limit=2)

    assert top_two == [("alpha", 5), ("beta", 3)]
    assert top_two is not result.items  # возвращается копия


@pytest.mark.unit
def test_word_frequency_result_top_without_limit():
    result = WordFrequencyResult(items=[("alpha", 5), ("beta", 3)])

    all_items = result.top()
    assert all_items == result.items
    assert all_items is not result.items


@pytest.mark.unit
def test_word_frequency_result_to_text():
    result = WordFrequencyResult(items=[("alpha", 5), ("beta", 3)])

    text = result.to_text()

    assert text == "alpha: 5\nbeta: 3"

