"""Сервис для анализа слов."""

from collections import Counter
from typing import Iterable, List

import pymorphy3

from app.domain.models.word_analysis import WordAnalysisConfig, WordFrequencyResult
from app.utils.text_analysis import WORD_PATTERN, TIMESTAMP_PATTERN, POS_TO_EXCLUDE


class WordAnalysisService:
    """Чистая бизнес-логика анализа слов (без чтения файлов)."""

    def __init__(self, morph_analyzer: pymorphy3.MorphAnalyzer):
        self._morph = morph_analyzer

    def extract_text(self, lines: Iterable[str]) -> str:
        text_lines: List[str] = []
        for line in lines:
            stripped = line.strip()
            match = TIMESTAMP_PATTERN.match(stripped)
            if match:
                remainder = stripped[match.end():].strip()
                if remainder:
                    text_lines.append(remainder)
            elif stripped:
                text_lines.append(stripped)
        return " ".join(text_lines).lower()

    def extract_words(self, text: str) -> List[str]:
        return WORD_PATTERN.findall(text)

    def lemmatize_and_filter(self, words: List[str], config: WordAnalysisConfig) -> List[str]:
        if not config.lemmatize:
            return words

        filtered: List[str] = []
        for word in words:
            parsed = self._morph.parse(word)[0]
            if parsed.tag.POS in POS_TO_EXCLUDE:
                continue
            if config.exclude_names and "Name" in parsed.tag.grammemes:
                continue
            filtered.append(parsed.normal_form)
        return filtered

    def filter_stopwords(self, words: List[str], stopwords: Iterable[str]) -> List[str]:
        stop_set = set(stopwords) if stopwords else set()
        if not stop_set:
            return words
        return [word for word in words if word not in stop_set]

    def analyze(
        self,
        lines: Iterable[str],
        stopwords: Iterable[str],
        config: WordAnalysisConfig,
    ) -> WordFrequencyResult:
        text = self.extract_text(lines)
        words = self.extract_words(text)
        if not words:
            raise ValueError("Не найдено слов длиной >= 3 символов")

        words = self.lemmatize_and_filter(words, config)
        words = self.filter_stopwords(words, stopwords)

        counts = Counter(words).most_common(config.limit)
        if not counts:
            raise ValueError("После фильтрации не осталось слов для анализа")

        return WordFrequencyResult(items=counts)

