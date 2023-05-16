from typing import List

import time
from minio import Minio
from urllib3.exceptions import MaxRetryError
from core.config import get_app_settings
import os

settings = get_app_settings()


# global minio_client
minio_client = Minio(
    endpoint=settings.minio_endpoint,
    access_key=settings.minio_access_key,
    secret_key=settings.minio_secret_key, secure=False)

try:
    video_bucket = "iva-video"
    found = minio_client.bucket_exists(video_bucket)
    if not found:
        minio_client.make_bucket(video_bucket)
except MaxRetryError:
    print("Minio Connection Refused")
except Exception as e:
    print(e)


def upload_video(video_path, object_name):
    is_remove = True
    try:
        minio_client.fput_object(
            bucket_name="iva-video", object_name=object_name, file_path=video_path, content_type="audio/mp4"
        )
        print(f"MinIO upload success with local video_path: {video_path} and object_name: {object_name}")
        is_remove = True
    except Exception as ex:
        print(f"MinIO upload video err {ex.__class__}, video_path: {video_path}, object_name: {object_name}")

    if is_remove:
        try:
            # remove file video in local :(.
            os.remove(video_path)
            print(f"Success remove local video: {video_path}")

        except Exception as ex:
            print(f"Cannot remove local video, err {ex.__class__}")
