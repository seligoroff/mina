"""Сервис для анализа слов из транскрипций.

Сервис выполняет частотный анализ слов из текста транскрипции:
- Извлечение текста с удалением таймкодов
- Лемматизация (опционально)
- Фильтрация по стоп-словам
- Частотный анализ
"""

from collections import Counter
from typing import List, Tuple, Optional, Set
import pymorphy3
from app.utils.text_analysis import WORD_PATTERN, TIMESTAMP_PATTERN, POS_TO_EXCLUDE


class WordAnalysisService:
    """Сервис для анализа частоты слов в транскрипциях."""
    
    def __init__(self):
        """Инициализация сервиса."""
        self._morph = None
    
    def load_stopwords(self, stopwords_path: Optional[str]) -> Set[str]:
        """Загружает стоп-слова из файла.
        
        Args:
            stopwords_path: Путь к файлу со стоп-словами (по одному слову на строку)
        
        Returns:
            Множество стоп-слов в нижнем регистре
        
        Raises:
            FileNotFoundError: Если файл не найден
            IOError: Если произошла ошибка при чтении файла
        """
        if not stopwords_path:
            return set()
        
        try:
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                return {line.strip().lower() for line in f if line.strip()}
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл со стоп-словами не найден: {stopwords_path}")
        except Exception as e:
            raise IOError(f"Ошибка при чтении файла со стоп-словами {stopwords_path}: {e}")
    
    def read_transcript_file(self, file_path: str) -> List[str]:
        """Читает файл с транскрипцией.
        
        Args:
            file_path: Путь к файлу с транскрипцией
        
        Returns:
            Список строк из файла
        
        Raises:
            FileNotFoundError: Если файл не найден
            IOError: Если произошла ошибка при чтении файла
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        except Exception as e:
            raise IOError(f"Ошибка при чтении файла {file_path}: {e}")
    
    def extract_text_from_lines(self, lines: List[str]) -> str:
        """Извлекает текст из строк, убирая таймкоды.
        
        Args:
            lines: Список строк из файла транскрипции
        
        Returns:
            Полный текст в нижнем регистре, собранный из всех строк
        """
        text_lines = []
        for line in lines:
            stripped = line.strip()
            # Проверяем, есть ли таймкод в начале строки
            match = TIMESTAMP_PATTERN.match(stripped)
            if match:
                # Извлекаем текст после таймкода
                text_after_timestamp = stripped[match.end():].strip()
                if text_after_timestamp:
                    text_lines.append(text_after_timestamp)
            else:
                # Если таймкода нет, берем всю строку
                if stripped:
                    text_lines.append(stripped)
        
        return ' '.join(text_lines).lower()
    
    def extract_words(self, text: str) -> List[str]:
        """Извлекает слова из текста (длиной >= 3).
        
        Args:
            text: Текст для анализа
        
        Returns:
            Список найденных слов
        """
        return WORD_PATTERN.findall(text)
    
    def _get_morph_analyzer(self):
        """Ленивая инициализация морфологического анализатора."""
        if self._morph is None:
            self._morph = pymorphy3.MorphAnalyzer(lang='ru')
        return self._morph
    
    def lemmatize_and_filter(self, 
                             words: List[str], 
                             lemmatize: bool,
                             no_names: bool) -> List[str]:
        """Выполняет лемматизацию и грамматическую фильтрацию слов.
        
        Args:
            words: Список слов для обработки
            lemmatize: Флаг включения лемматизации
            no_names: Флаг исключения имен собственных
        
        Returns:
            Отфильтрованный список слов (лемм)
        """
        if not lemmatize:
            return words
        
        morph = self._get_morph_analyzer()
        filtered = []
        
        for word in words:
            parsed = morph.parse(word)[0]
            lemma = parsed.normal_form
            
            # Исключаем определенные части речи
            if parsed.tag.POS in POS_TO_EXCLUDE:
                continue
            
            # Исключаем имена собственные, если указано
            if no_names and 'Name' in parsed.tag.grammemes:
                continue
            
            filtered.append(lemma)
        
        return filtered
    
    def filter_stopwords(self, words: List[str], stopwords: Set[str]) -> List[str]:
        """Удаляет стоп-слова из списка слов.
        
        Args:
            words: Список слов
            stopwords: Множество стоп-слов
        
        Returns:
            Отфильтрованный список слов без стоп-слов
        """
        if not stopwords:
            return words
        
        return [word for word in words if word not in stopwords]
    
    def analyze_word_frequency(self, 
                               file_path: str,
                               stopwords: Optional[str] = None,
                               lemmatize: bool = False,
                               no_names: bool = False,
                               limit: int = 50) -> List[Tuple[str, int]]:
        """Выполняет полный цикл анализа частоты слов.
        
        Args:
            file_path: Путь к файлу с транскрипцией
            stopwords: Путь к файлу со стоп-словами (опционально)
            lemmatize: Флаг включения лемматизации
            no_names: Флаг исключения имен собственных
            limit: Максимальное количество слов в результате
        
        Returns:
            Список кортежей (слово, количество) отсортированных по частоте
        
        Raises:
            FileNotFoundError: Если файл транскрипции или стоп-слов не найден
            IOError: Если произошла ошибка при чтении файла
            ValueError: Если после фильтрации не осталось слов
        """
        # Загружаем стоп-слова
        stopword_set = self.load_stopwords(stopwords)
        
        # Читаем файл
        lines = self.read_transcript_file(file_path)
        
        # Извлекаем текст
        full_text = self.extract_text_from_lines(lines)
        
        # Извлекаем слова
        words = self.extract_words(full_text)
        
        if not words:
            raise ValueError(f"В файле {file_path} не найдено слов (длиной >= 3 символов)")
        
        # Лемматизация и грамматическая фильтрация
        words = self.lemmatize_and_filter(words, lemmatize, no_names)
        
        # Удаляем стоп-слова
        words = self.filter_stopwords(words, stopword_set)
        
        # Частотный анализ
        word_counts = Counter(words)
        most_common = word_counts.most_common(limit)
        
        if not most_common:
            raise ValueError("После фильтрации не осталось слов для анализа")
        
        return most_common
    
    def format_result(self, word_frequency: List[Tuple[str, int]]) -> str:
        """Форматирует результат анализа в строку.
        
        Args:
            word_frequency: Список кортежей (слово, количество)
        
        Returns:
            Отформатированная строка с результатами
        """
        return '\n'.join(f'{word}: {count}' for word, count in word_frequency)

