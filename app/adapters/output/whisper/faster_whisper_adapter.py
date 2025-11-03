"""Адаптер для faster-whisper."""

from typing import Iterator, Any
from app.application.ports import ITranscriptionEngine
from app.domain.models.transcript import Segment


class FasterWhisperAdapter(ITranscriptionEngine):
    """Адаптер для faster-whisper.
    
    Оборачивает библиотеку faster-whisper и адаптирует её API
    к нашему интерфейсу ITranscriptionEngine.
    """
    
    def __init__(self, faster_whisper_model_class, compute_type: str = 'int8'):
        """
        Args:
            faster_whisper_model_class: Класс WhisperModel из faster_whisper
            compute_type: Тип вычислений ('int8', 'float16', 'float32')
        """
        self._faster_whisper_model_class = faster_whisper_model_class
        self._compute_type = compute_type
    
    def load_model(self, model_name: str, **kwargs) -> Any:
        """Загружает модель faster-whisper.
        
        Args:
            model_name: Название модели (например, 'base', 'small', 'medium')
            **kwargs: Дополнительные параметры (игнорируются)
        
        Returns:
            Загруженная модель FasterWhisper (WhisperModel)
        """
        return self._faster_whisper_model_class(model_name, compute_type=self._compute_type)
    
    def transcribe(self,
                   model: Any,
                   audio_path: str,
                   language: str,
                   **kwargs) -> Iterator[Segment]:
        """Выполняет транскрипцию аудиофайла через faster-whisper.
        
        Args:
            model: Загруженная модель FasterWhisper (результат load_model)
            audio_path: Путь к аудиофайлу
            language: Код языка транскрипции (ISO 639-1, например 'ru', 'en')
            **kwargs: Дополнительные параметры (beam_size и т.д.)
        
        Yields:
            Segment: Сегменты транскрипции с таймингами
        """
        beam_size = kwargs.get('beam_size', 5)
        verbose = kwargs.get('verbose', False)
        
        # Выполняем транскрипцию через faster-whisper
        # Для длинных видео используем оптимизированные параметры:
        # - condition_on_previous_text=False - уменьшает использование памяти
        # - word_timestamps=False - не нужно хранить временные метки слов (только сегментов)
        # - vad_filter=True - фильтрация голосовой активности для более эффективной обработки
        segments, info = model.transcribe(
            audio_path, 
            beam_size=beam_size, 
            language=language,
            condition_on_previous_text=False,  # Экономит память для длинных видео
            word_timestamps=False,  # Не нужны временные метки слов, только сегментов
            vad_filter=True,  # Используем VAD для более эффективной обработки длинных видео
            vad_parameters=dict(
                min_silence_duration_ms=500,
                threshold=0.5
            )
        )
        
        if verbose:
            import sys
            print(f"Faster-whisper: обнаружен язык '{info.language}' (вероятность: {info.language_probability:.2f})", 
                  file=sys.stderr)
        
        # Конвертируем объекты faster-whisper в доменные модели Segment
        segment_count = 0
        last_end_time = 0.0
        generator_finished = False
        
        try:
            for segment_obj in segments:
                segment_count += 1
                try:
                    segment = Segment(
                        start=segment_obj.start,
                        end=segment_obj.end,
                        text=segment_obj.text.strip()
                    )
                    
                    # Проверяем, что сегменты идут последовательно
                    if segment.start < last_end_time and last_end_time > 0:
                        if verbose:
                            import sys
                            print(f"Предупреждение: сегмент {segment_count} начинается раньше предыдущего "
                                  f"({segment.start:.2f} < {last_end_time:.2f})", file=sys.stderr)
                    
                    last_end_time = segment.end
                    yield segment
                    
                    # Логируем прогресс каждые 500 сегментов или каждые 10 минут
                    if verbose:
                        import sys
                        if segment_count % 500 == 0:
                            print(f"Faster-whisper: обработано {segment_count} сегментов, "
                                  f"время: {last_end_time/60:.1f} мин", file=sys.stderr)
                    
                except Exception as e:
                    # Логируем ошибку при обработке одного сегмента, но продолжаем
                    import sys
                    print(f"Ошибка при обработке сегмента {segment_count}: {e}", file=sys.stderr)
                    continue
            
            # Если цикл завершился без исключения, генератор дошел до конца
            generator_finished = True
                    
        except StopIteration:
            # Генератор завершился нормально (это нормально для итераторов)
            generator_finished = True
        except GeneratorExit:
            # Генератор был закрыт принудительно
            import sys
            print(f"ПРЕДУПРЕЖДЕНИЕ: Генератор faster-whisper был закрыт принудительно "
                  f"(обработано {segment_count} сегментов, последнее время: {last_end_time:.2f} сек)", 
                  file=sys.stderr)
            generator_finished = False
            raise
        except Exception as e:
            # Логируем критическую ошибку
            import sys
            print(f"Критическая ошибка в генераторе faster-whisper "
                  f"(обработано {segment_count} сегментов, последнее время: {last_end_time:.2f}): {e}", 
                  file=sys.stderr)
            generator_finished = False
            raise
        finally:
            if verbose or not generator_finished:
                import sys
                print(f"Faster-whisper: обработано {segment_count} сегментов, "
                      f"общая длительность: {last_end_time:.2f} сек ({last_end_time/60:.1f} мин)", 
                      file=sys.stderr)
                
                # Проверяем, не оборвалась ли транскрипция подозрительно рано
                # Если генератор завершился "нормально", но на времени меньше 40 минут,
                # это может быть проблемой faster-whisper с длинными видео
                if generator_finished and last_end_time > 0:
                    # Если последний сегмент меньше 45 минут, а сегментов обработано менее 2000,
                    # это может быть проблема обрыва (особенно для длинных видео 1+ часа)
                    if last_end_time < 2700 and segment_count < 1000:
                        print(f"\n⚠️  ПРЕДУПРЕЖДЕНИЕ: Возможна проблема с faster-whisper! "
                              f"Транскрипция завершилась на {last_end_time/60:.1f} минуте "
                              f"({segment_count} сегментов).", 
                              file=sys.stderr)
                        print(f"⚠️  Для длинных видео (1+ час) faster-whisper может обрывать транскрипцию раньше времени.", 
                              file=sys.stderr)
                        print(f"⚠️  Рекомендации: попробуйте использовать модель smaller (small вместо medium) "
                              f"или разбейте видео на части.", 
                              file=sys.stderr)
                
                if not generator_finished and last_end_time > 0:
                    print(f"ПРЕДУПРЕЖДЕНИЕ: Генератор faster-whisper не завершился нормально. "
                          f"Последний сегмент на {last_end_time/60:.1f} минуте.", 
                          file=sys.stderr)

