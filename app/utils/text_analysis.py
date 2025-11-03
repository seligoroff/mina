"""Утилиты для анализа текста транскрипций."""

import re

# Регулярные выражения для извлечения слов и таймкодов
WORD_PATTERN = re.compile(r'\b[а-яА-ЯёЁa-zA-Z]{3,}\b', re.UNICODE)
TIMESTAMP_PATTERN = re.compile(r'^\[\d{1,3}\.\d{2}\s*-\s*\d{1,3}\.\d{2}\]')

# Части речи, которые считаем шумом (для исключения при лемматизации)
POS_TO_EXCLUDE = {'NPRO', 'ADVB', 'PRCL', 'CONJ', 'PREP', 'INTJ'}



