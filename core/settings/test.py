import logging

from core.settings.app import AppSettings


class TestAppSettings(AppSettings):
    debug: bool = True

    logging_level: int = logging.DEBUG

    class Config(AppSettings.Config):
        env_file = "test.env"