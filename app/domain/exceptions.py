"""Доменные исключения приложения."""


class DomainError(Exception):
    """Базовое доменное исключение."""


class ProtocolClientError(DomainError):
    """Ошибка, возникшая при обращении к провайдеру LLM."""


