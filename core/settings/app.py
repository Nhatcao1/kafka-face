import logging

from pydantic import PostgresDsn

from core.settings.base import BaseAppSettings


class AppSettings(BaseAppSettings):
    debug: bool = False

    database_url: PostgresDsn = "postgresql://postgres:changeme@localhost:5432/face_management"
    max_connection_count: int = 10
    min_connection_count: int = 10

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minio"
    minio_secret_key: str = "minio123"
    minio_inference_bucket_name: str = "inference-data"
    minio_event_bucket_name: str = "event-image"
    minio_video_bucket_name: str = "video"

    triton_host: str = "localhost:8001"
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_index: str = "flat"
    face_registration_bucket_name: str = "registration-faces"

    # detection config
    detection_threshold: float = 50
    nms_threshold: float = 0.4
    detection_max_size: int = 600
    enable_anti_spoofing: bool = True

    # recognition config
    recognition_model_type = "mxnet"
    feature_vector_dimension: int = 512
    distance_threshold: float = 0.78
    recognition_threshold: float = 65.0

    video_duration: int = 10
    fps: int = 4

    logging_level: int = logging.INFO

    class Config:
        validate_assignment = True
