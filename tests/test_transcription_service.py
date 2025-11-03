"""Тесты для TranscriptionService."""

import pytest
from unittest.mock import Mock, patch
from app.application.services import TranscriptionService
from app.application.ports import ITranscriptionEngine, ITranscriptSegmentWriter
from app.domain.models.transcript import Segment
from app.adapters.output.file_writer import FileOutputWriter


@pytest.mark.unit
class TestTranscriptionService:
    """Тесты для TranscriptionService."""
    
    def test_transcription_service_init(self):
        """Тест инициализации сервиса."""
        mock_engine = Mock(spec=ITranscriptionEngine)
        service = TranscriptionService(engine=mock_engine)
        
        assert service._engine is mock_engine
    
    def test_transcription_service_transcribe_writes_segments(self, tmp_path):
        """Тест записи сегментов через output_writer."""
        mock_engine = Mock(spec=ITranscriptionEngine)
        mock_model = Mock()
        
        segments = [
            Segment(start=0.0, end=1.0, text="Первое"),
            Segment(start=1.0, end=2.0, text="Второе"),
        ]
        mock_engine.load_model.return_value = mock_model
        mock_engine.transcribe.return_value = iter(segments)
        
        service = TranscriptionService(engine=mock_engine)
        output_file = tmp_path / "output.txt"
        output_writer = FileOutputWriter(str(output_file), verbose=False)
        
        result = list(service.transcribe(
            input_path="test.mp3",
            output_writer=output_writer,
            model_name="small",
            language="ru"
        ))
        
        # Проверяем, что writer был закрыт
        assert output_writer._file.closed
        
        # Проверяем запись в файл
        content = output_file.read_text(encoding='utf-8')
        assert "Первое" in content
        assert "Второе" in content
        
        # Проверяем возврат сегментов
        assert len(result) == 2
        assert result[0].text == "Первое"
        assert result[1].text == "Второе"
        
        # Проверяем вызовы методов engine
        mock_engine.load_model.assert_called_once_with("small")
        mock_engine.transcribe.assert_called_once()
    
    def test_transcription_service_transcribe_calls_engine_with_correct_params(self, tmp_path):
        """Тест вызова transcribe() у engine с правильными параметрами."""
        mock_engine = Mock(spec=ITranscriptionEngine)
        mock_model = Mock()
        
        segments = [Segment(start=0.0, end=1.0, text="Тест")]
        mock_engine.load_model.return_value = mock_model
        mock_engine.transcribe.return_value = iter(segments)
        
        service = TranscriptionService(engine=mock_engine)
        output_file = tmp_path / "output.txt"
        output_writer = FileOutputWriter(str(output_file), verbose=False)
        
        list(service.transcribe(
            input_path="test.mp3",
            output_writer=output_writer,
            model_name="base",
            language="en",
            beam_size=10,
            verbose=True
        ))
        
        # Проверяем вызов load_model
        mock_engine.load_model.assert_called_once_with("base")
        
        # Проверяем вызов transcribe с правильными параметрами
        mock_engine.transcribe.assert_called_once_with(
            model=mock_model,
            audio_path="test.mp3",
            language="en",
            beam_size=10,
            verbose=True
        )
    
    def test_transcription_service_transcribe_default_params(self, tmp_path):
        """Тест использования параметров по умолчанию."""
        mock_engine = Mock(spec=ITranscriptionEngine)
        mock_model = Mock()
        
        segments = [Segment(start=0.0, end=1.0, text="Тест")]
        mock_engine.load_model.return_value = mock_model
        mock_engine.transcribe.return_value = iter(segments)
        
        service = TranscriptionService(engine=mock_engine)
        output_file = tmp_path / "output.txt"
        output_writer = FileOutputWriter(str(output_file), verbose=False)
        
        list(service.transcribe(
            input_path="test.mp3",
            output_writer=output_writer,
            model_name="small"
        ))
        
        # Проверяем вызов transcribe с параметрами по умолчанию
        mock_engine.transcribe.assert_called_once_with(
            model=mock_model,
            audio_path="test.mp3",
            language="ru",
            beam_size=5,
            verbose=False
        )
    
    def test_transcription_service_transcribe_always_closes_writer(self, tmp_path):
        """Тест закрытия writer даже при ошибке во время записи."""
        mock_engine = Mock(spec=ITranscriptionEngine)
        mock_model = Mock()
        mock_engine.load_model.return_value = mock_model
        
        # Создаем генератор, который выбрасывает ошибку после первого элемента
        # Это имитирует ошибку во время итерации по сегментам
        def error_generator():
            yield Segment(start=0.0, end=1.0, text="Первый")
            raise ValueError("Ошибка транскрипции")
        
        mock_engine.transcribe.return_value = error_generator()
        
        service = TranscriptionService(engine=mock_engine)
        output_file = tmp_path / "output.txt"
        output_writer = FileOutputWriter(str(output_file), verbose=False)
        
        with pytest.raises(ValueError, match="Ошибка транскрипции"):
            list(service.transcribe(
                input_path="test.mp3",
                output_writer=output_writer,
                model_name="small"
            ))
        
        # Writer должен быть закрыт даже при ошибке (благодаря try-finally)
        assert output_writer._file.closed
        
        # Проверяем, что хотя бы первый сегмент был записан
        content = output_file.read_text(encoding='utf-8')
        assert "Первый" in content
    
    def test_transcription_service_transcribe_closes_writer_on_engine_error(self, tmp_path):
        """Тест закрытия writer при ошибке в engine.transcribe()."""
        mock_engine = Mock(spec=ITranscriptionEngine)
        mock_model = Mock()
        mock_engine.load_model.return_value = mock_model
        
        # Создаем генератор, который выбрасывает ошибку после первого элемента
        def error_generator():
            yield Segment(start=0.0, end=1.0, text="Первый")
            raise RuntimeError("Ошибка в engine")
        
        mock_engine.transcribe.return_value = error_generator()
        
        service = TranscriptionService(engine=mock_engine)
        output_file = tmp_path / "output.txt"
        output_writer = FileOutputWriter(str(output_file), verbose=False)
        
        with pytest.raises(RuntimeError, match="Ошибка в engine"):
            list(service.transcribe(
                input_path="test.mp3",
                output_writer=output_writer,
                model_name="small"
            ))
        
        # Writer должен быть закрыт даже при ошибке
        assert output_writer._file.closed
    
    def test_transcription_service_transcribe_returns_iterator(self, tmp_path):
        """Тест возврата итератора сегментов."""
        mock_engine = Mock(spec=ITranscriptionEngine)
        mock_model = Mock()
        
        segments = [
            Segment(start=0.0, end=1.0, text="Первый"),
            Segment(start=1.0, end=2.0, text="Второй"),
            Segment(start=2.0, end=3.0, text="Третий"),
        ]
        mock_engine.load_model.return_value = mock_model
        mock_engine.transcribe.return_value = iter(segments)
        
        service = TranscriptionService(engine=mock_engine)
        output_file = tmp_path / "output.txt"
        output_writer = FileOutputWriter(str(output_file), verbose=False)
        
        result = service.transcribe(
            input_path="test.mp3",
            output_writer=output_writer,
            model_name="small"
        )
        
        # Проверяем, что результат - итератор
        assert hasattr(result, '__iter__')
        
        # Преобразуем в список и проверяем содержимое
        result_list = list(result)
        assert len(result_list) == 3
        assert result_list[0].text == "Первый"
        assert result_list[1].text == "Второй"
        assert result_list[2].text == "Третий"
    
    def test_transcription_service_transcribe_empty_segments(self, tmp_path):
        """Тест обработки пустого списка сегментов."""
        mock_engine = Mock(spec=ITranscriptionEngine)
        mock_model = Mock()
        
        mock_engine.load_model.return_value = mock_model
        mock_engine.transcribe.return_value = iter([])
        
        service = TranscriptionService(engine=mock_engine)
        output_file = tmp_path / "output.txt"
        output_writer = FileOutputWriter(str(output_file), verbose=False)
        
        result = list(service.transcribe(
            input_path="test.mp3",
            output_writer=output_writer,
            model_name="small"
        ))
        
        # Проверяем, что результат пуст
        assert len(result) == 0
        
        # Проверяем, что writer был закрыт
        assert output_writer._file.closed
    
    def test_transcription_service_transcribe_with_mock_writer(self, tmp_path):
        """Тест записи через мокированный output_writer."""
        mock_engine = Mock(spec=ITranscriptionEngine)
        mock_writer = Mock(spec=ITranscriptSegmentWriter)
        mock_model = Mock()
        
        segments = [
            Segment(start=0.0, end=1.0, text="Тест"),
        ]
        mock_engine.load_model.return_value = mock_model
        mock_engine.transcribe.return_value = iter(segments)
        
        service = TranscriptionService(engine=mock_engine)
        
        result = list(service.transcribe(
            input_path="test.mp3",
            output_writer=mock_writer,
            model_name="small"
        ))
        
        # Проверяем, что write_segment был вызван
        mock_writer.write_segment.assert_called_once_with(segments[0])
        
        # Проверяем, что close был вызван
        mock_writer.close.assert_called_once()
        
        # Проверяем возврат сегментов
        assert len(result) == 1
    
    def test_transcription_service_transcribe_writes_all_segments(self, tmp_path):
        """Тест записи всех сегментов через output_writer."""
        mock_engine = Mock(spec=ITranscriptionEngine)
        mock_writer = Mock(spec=ITranscriptSegmentWriter)
        mock_model = Mock()
        
        segments = [
            Segment(start=0.0, end=1.0, text="Первый"),
            Segment(start=1.0, end=2.0, text="Второй"),
            Segment(start=2.0, end=3.0, text="Третий"),
        ]
        mock_engine.load_model.return_value = mock_model
        mock_engine.transcribe.return_value = iter(segments)
        
        service = TranscriptionService(engine=mock_engine)
        
        result = list(service.transcribe(
            input_path="test.mp3",
            output_writer=mock_writer,
            model_name="small"
        ))
        
        # Проверяем, что write_segment был вызван для каждого сегмента
        assert mock_writer.write_segment.call_count == 3
        mock_writer.write_segment.assert_any_call(segments[0])
        mock_writer.write_segment.assert_any_call(segments[1])
        mock_writer.write_segment.assert_any_call(segments[2])
        
        # Проверяем, что close был вызван один раз
        mock_writer.close.assert_called_once()
        
        # Проверяем возврат всех сегментов
        assert len(result) == 3
    
    @patch('shutil.which')
    def test_transcription_service_requires_ffmpeg(self, mock_which, tmp_path):
        """Тест проверки ffmpeg через декоратор."""
        mock_engine = Mock(spec=ITranscriptionEngine)
        service = TranscriptionService(engine=mock_engine)
        output_file = tmp_path / "output.txt"
        output_writer = FileOutputWriter(str(output_file), verbose=False)
        
        # Мокируем отсутствие ffmpeg
        mock_which.return_value = None
        
        with pytest.raises(RuntimeError, match="ffmpeg не найден"):
            list(service.transcribe(
                input_path="test.mp3",
                output_writer=output_writer,
                model_name="small"
            ))
        
        # Проверяем, что load_model не был вызван (декоратор сработал раньше)
        mock_engine.load_model.assert_not_called()
        
        # Writer должен быть закрыт (если был создан)
        if output_writer._file:
            # Если файл был открыт до ошибки
            pass
    
    @patch('shutil.which')
    def test_transcription_service_with_ffmpeg_available(self, mock_which, tmp_path):
        """Тест выполнения транскрипции, когда ffmpeg доступен."""
        mock_engine = Mock(spec=ITranscriptionEngine)
        mock_model = Mock()
        
        segments = [Segment(start=0.0, end=1.0, text="Тест")]
        mock_engine.load_model.return_value = mock_model
        mock_engine.transcribe.return_value = iter(segments)
        
        service = TranscriptionService(engine=mock_engine)
        output_file = tmp_path / "output.txt"
        output_writer = FileOutputWriter(str(output_file), verbose=False)
        
        # Мокируем наличие ffmpeg
        mock_which.return_value = '/usr/bin/ffmpeg'
        
        result = list(service.transcribe(
            input_path="test.mp3",
            output_writer=output_writer,
            model_name="small"
        ))
        
        # Проверяем, что транскрипция прошла успешно
        assert len(result) == 1
        assert result[0].text == "Тест"
        
        # Проверяем, что load_model был вызван
        mock_engine.load_model.assert_called_once()
    
    def test_transcription_service_transcribe_with_verbose(self, tmp_path, capsys):
        """Тест транскрипции с verbose=True."""
        mock_engine = Mock(spec=ITranscriptionEngine)
        mock_model = Mock()
        
        segments = [Segment(start=0.0, end=1.0, text="Тест")]
        mock_engine.load_model.return_value = mock_model
        mock_engine.transcribe.return_value = iter(segments)
        
        service = TranscriptionService(engine=mock_engine)
        output_file = tmp_path / "output.txt"
        output_writer = FileOutputWriter(str(output_file), verbose=True)
        
        list(service.transcribe(
            input_path="test.mp3",
            output_writer=output_writer,
            model_name="small",
            verbose=True
        ))
        
        # Проверяем, что verbose был передан в engine.transcribe
        mock_engine.transcribe.assert_called_once()
        call_kwargs = mock_engine.transcribe.call_args[1]
        assert call_kwargs['verbose'] is True
    
    def test_transcription_service_transcribe_with_different_beam_sizes(self, tmp_path):
        """Тест транскрипции с различными значениями beam_size."""
        mock_engine = Mock(spec=ITranscriptionEngine)
        mock_model = Mock()
        
        segments = [Segment(start=0.0, end=1.0, text="Тест")]
        mock_engine.load_model.return_value = mock_model
        mock_engine.transcribe.return_value = iter(segments)
        
        service = TranscriptionService(engine=mock_engine)
        output_file = tmp_path / "output.txt"
        
        beam_sizes = [1, 5, 10, 20]
        
        for beam_size in beam_sizes:
            output_writer = FileOutputWriter(str(output_file), verbose=False)
            mock_engine.reset_mock()
            
            list(service.transcribe(
                input_path="test.mp3",
                output_writer=output_writer,
                model_name="small",
                beam_size=beam_size
            ))
            
            # Проверяем, что beam_size был передан в engine.transcribe
            call_kwargs = mock_engine.transcribe.call_args[1]
            assert call_kwargs['beam_size'] == beam_size
    
    def test_transcription_service_transcribe_with_different_languages(self, tmp_path):
        """Тест транскрипции с различными языками."""
        mock_engine = Mock(spec=ITranscriptionEngine)
        mock_model = Mock()
        
        segments = [Segment(start=0.0, end=1.0, text="Test")]
        mock_engine.load_model.return_value = mock_model
        mock_engine.transcribe.return_value = iter(segments)
        
        service = TranscriptionService(engine=mock_engine)
        output_file = tmp_path / "output.txt"
        
        languages = ['ru', 'en', 'es', 'de', 'fr']
        
        for language in languages:
            output_writer = FileOutputWriter(str(output_file), verbose=False)
            mock_engine.reset_mock()
            
            list(service.transcribe(
                input_path="test.mp3",
                output_writer=output_writer,
                model_name="small",
                language=language
            ))
            
            # Проверяем, что language был передан в engine.transcribe
            call_kwargs = mock_engine.transcribe.call_args[1]
            assert call_kwargs['language'] == language

