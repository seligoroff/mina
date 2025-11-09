"""Тесты для app.main.create_app()."""

from unittest.mock import MagicMock, patch


def test_create_app_returns_expected_structure():
    with patch("pymorphy3.MorphAnalyzer"), \
         patch("app.adapters.input.cli.ScribeCommandHandler"), \
         patch("app.adapters.input.cli.ProtocolCommandHandler"), \
         patch("app.adapters.input.cli.TagCommandHandler"):
        from app.main import create_app
        app = create_app()

    assert set(app.keys()) == {"handlers", "services", "factories"}
    handlers = app["handlers"]
    assert set(handlers.keys()) == {"scribe", "tag", "protocol"}
    services = app["services"]
    assert set(services.keys()) == {"word_analysis"}
    factories = app["factories"]
    assert {"transcription", "transcription_service", "protocol_client",
            "protocol_service", "word_analysis_service"} <= set(factories.keys())


def test_create_app_tag_handler_uses_shared_morph():
    morph_instance = MagicMock()
    tag_handler_instance = MagicMock()
    service_instance_1 = MagicMock()
    service_instance_2 = MagicMock()
    service_instance_1._morph = morph_instance
    service_instance_2._morph = morph_instance

    with patch("pymorphy3.MorphAnalyzer", return_value=morph_instance), \
         patch("app.adapters.input.cli.TagCommandHandler", return_value=tag_handler_instance) as tag_handler_mock, \
         patch("app.factories.create_word_analysis_service") as word_service_mock:
        word_service_mock.side_effect = [service_instance_1, service_instance_2]

        from app.main import create_app

        app = create_app()

    handlers = app["handlers"]
    services = app["services"]

    assert handlers["tag"] is tag_handler_instance
    assert services["word_analysis"] is service_instance_1

    analysis_factory = tag_handler_mock.call_args.kwargs["analysis_service_factory"]
    derived_service = analysis_factory()
    assert derived_service is service_instance_2

    first_call_kwargs = word_service_mock.call_args_list[0].kwargs
    second_call_kwargs = word_service_mock.call_args_list[1].kwargs
    assert first_call_kwargs["dependencies"]["morph"] is morph_instance
    assert second_call_kwargs["dependencies"]["morph"] is morph_instance

