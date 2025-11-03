"""Сервис транскрипции аудио.

Сервис транскрипции, работающий через абстракции ITranscriptionEngine и ITranscriptSegmentWriter.
Использует доменные модели Segment для работы с результатами транскрипции.
"""

from typing import Iterator
from app.application.ports import ITranscriptionEngine, ITranscriptSegmentWriter
from app.domain.models.transcript import Segment
from app.utils.decorators import require_ffmpeg


class TranscriptionService:
    """Сервис транскрипции аудио.
    
    Работает через абстракции ITranscriptionEngine и ITranscriptSegmentWriter, не зная
    конкретные реализации (Whisper, faster-whisper, файловый вывод и т.д.).
    Использует доменные модели Segment.
    """
    
    def __init__(self, engine: ITranscriptionEngine):
        """
        Args:
            engine: Адаптер движка транскрипции, реализующий ITranscriptionEngine
        """
        self._engine = engine
    
    @require_ffmpeg
    def transcribe(self,
                   input_path: str,
                   output_writer: ITranscriptSegmentWriter,
                   model_name: str,
                   language: str = 'ru',
                   **kwargs) -> Iterator[Segment]:
        """Выполняет транскрипцию аудиофайла.
        
        Args:
            input_path: Путь к аудиофайлу
            output_writer: Адаптер для записи сегментов транскрипции (ITranscriptSegmentWriter)
            model_name: Название модели (например, 'base', 'small', 'medium')
            language: Код языка транскрипции (ISO 639-1, например 'ru', 'en')
            **kwargs: Дополнительные параметры (beam_size и т.д.)
        
        Yields:
            Segment: Сегменты транскрипции с таймингами
        
        Raises:
            RuntimeError: Если ffmpeg не найден (проверяется декоратором @require_ffmpeg)
        """
        # Загружаем модель через адаптер (compute_type уже настроен в адаптере)
        model = self._engine.load_model(model_name)
        
        # Выполняем транскрипцию через адаптер (получаем Iterator[Segment])
        segments = self._engine.transcribe(
            model=model,
            audio_path=input_path,
            language=language,
            beam_size=kwargs.get('beam_size', 5),
            verbose=kwargs.get('verbose', False)
        )
        
        # Записываем сегменты через адаптер вывода (I/O операции изолированы)
        segments_list = []
        segment_count = 0
        last_segment_time = 0.0
        verbose = kwargs.get('verbose', False)
        generator_completed_normally = False
        
        try:
            for segment in segments:
                segment_count += 1
                last_segment_time = max(last_segment_time, segment.end)
                
                # Логируем прогресс каждые 100 сегментов или каждые 10 минут
                if verbose:
                    import sys
                    if segment_count % 100 == 0 or segment_count == 1:
                        print(f"Обработано сегментов: {segment_count}, последнее время: {last_segment_time:.2f} сек ({last_segment_time/60:.1f} мин)", 
                              file=sys.stderr)
                    # Логируем каждые 10 минут для длинных видео
                    elif int(last_segment_time) % 600 == 0 and int(last_segment_time) > 0:
                        print(f"Прогресс: {last_segment_time/60:.1f} минут обработано, сегментов: {segment_count}", 
                              file=sys.stderr)
                
                try:
                    # Записываем сегмент через порт (не знаем, куда именно - файл, консоль, БД и т.д.)
                    output_writer.write_segment(segment)
                    segments_list.append(segment)
                except Exception as e:
                    # Логируем ошибку, но продолжаем обработку остальных сегментов
                    import sys
                    print(f"Ошибка при записи сегмента [{segment.start:.2f} - {segment.end:.2f}]: {e}", 
                          file=sys.stderr)
                    # Все равно добавляем сегмент в список для возврата
                    segments_list.append(segment)
            
            # Если цикл завершился без исключения, генератор дошел до конца
            generator_completed_normally = True
            
        except StopIteration:
            # Генератор завершился нормально (это нормально для итераторов)
            generator_completed_normally = True
        except GeneratorExit:
            # Генератор был закрыт принудительно
            import sys
            print(f"ПРЕДУПРЕЖДЕНИЕ: Генератор был закрыт принудительно "
                  f"(обработано {segment_count} сегментов, последнее время: {last_segment_time:.2f} сек)", 
                  file=sys.stderr)
            generator_completed_normally = False
        except Exception as e:
            # Логируем критическую ошибку при обработке генератора
            import sys
            print(f"Критическая ошибка при обработке сегментов "
                  f"(обработано {segment_count} сегментов, последнее время: {last_segment_time:.2f} сек): {e}", 
                  file=sys.stderr)
            generator_completed_normally = False
            raise
        finally:
            # Всегда закрываем writer, даже если произошла ошибка
            output_writer.close()
            # Логируем итоговую статистику
            import sys
            if verbose or not generator_completed_normally:
                print(f"Завершена транскрипция: обработано {len(segments_list)} сегментов, "
                      f"последнее время: {last_segment_time:.2f} сек ({last_segment_time/60:.1f} мин)", 
                      file=sys.stderr)
                
                # Проверяем, не оборвалась ли транскрипция подозрительно рано
                # Если генератор завершился "нормально", но на времени меньше 45 минут,
                # это может быть проблема faster-whisper с длинными видео
                if generator_completed_normally and last_segment_time > 0:
                    # Если последний сегмент меньше 45 минут, а сегментов обработано менее 1000,
                    # это может быть проблема обрыва (особенно для длинных видео 1+ часа)
                    if last_segment_time < 2700 and len(segments_list) < 1000:
                        print(f"\n⚠️  ПРЕДУПРЕЖДЕНИЕ: Возможна проблема с транскрипцией длинного видео! "
                              f"Транскрипция завершилась на {last_segment_time/60:.1f} минуте "
                              f"({len(segments_list)} сегментов).", 
                              file=sys.stderr)
                        print(f"⚠️  Для длинных видео (1+ час) faster-whisper может обрывать транскрипцию раньше времени.", 
                              file=sys.stderr)
                        print(f"⚠️  Рекомендации: попробуйте использовать модель smaller (small вместо medium) "
                              f"или разбейте видео на части.", 
                              file=sys.stderr)
                
                if not generator_completed_normally and last_segment_time > 0:
                    print(f"ПРЕДУПРЕЖДЕНИЕ: Генератор завершился раньше времени или с ошибкой. "
                          f"Последний сегмент на {last_segment_time/60:.1f} минуте. "
                          f"Возможно, транскрипция неполная.", 
                          file=sys.stderr)
        
        # Возвращаем итератор сегментов для дальнейшей обработки
        return iter(segments_list)

