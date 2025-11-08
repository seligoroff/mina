"""Фабрики для анализа слов."""

from typing import Dict, Optional

import pymorphy3

from app.application.services.word_analysis import WordAnalysisService


def create_word_analysis_service(
    dependencies: Optional[Dict[str, object]] = None,
) -> WordAnalysisService:
    deps = dependencies or {}
    morph = deps.get("morph") or pymorphy3.MorphAnalyzer(lang="ru")
    return WordAnalysisService(morph_analyzer=morph)

