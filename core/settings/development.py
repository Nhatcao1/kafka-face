import logging

from core.settings.app import AppSettings


class DevAppSettings(AppSettings):
    debug: bool = True
    database_url: str = "postgresql://postgres:changeme@10.61.184.31:5432/face_management"
    tensorflow_serving_host: str = "10.61.184.31:8500"
    triton_host: str = "10.61.184.31:8001"
    milvus_host: str = "10.61.184.31"
    milvus_port: int = 19530
    logging_level: int = logging.DEBUG

    class Config(AppSettings.Config):
        env_file = "dev.env"
