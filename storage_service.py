import asyncio
import io
import os
import time


import cv2
import numpy as np
from minio import Minio
from urllib3.exceptions import MaxRetryError

# from iva import worker
from core.config import get_app_settings
from log import logger
# from util.entity import Label, IVAConfig

settings = get_app_settings()


def get_minio_client():
    global minio_client
    time.sleep(1)
    print("Connecting to Minio Server")
    minio_client = Minio(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key, secure=False
    )


print("Connected to Minio Server")

get_minio_client()
try:
    inference_bucket_name = settings.minio_inference_bucket_name
    found = minio_client.bucket_exists(inference_bucket_name)
    if not found:
        minio_client.make_bucket(inference_bucket_name)

    event_bucket_name = settings.minio_event_bucket_name
    found = minio_client.bucket_exists(event_bucket_name)
    if not found:
        minio_client.make_bucket(event_bucket_name)
        
    video_bucket_name = settings.minio_video_bucket_name
    found = minio_client.bucket_exists(video_bucket_name)
    if not found:
        minio_client.make_bucket(video_bucket_name)
        
    found = minio_client.bucket_exists("registration-faces")
    if not found:
        minio_client.make_bucket("registration-faces")

    
except MaxRetryError:
    logger.error("Minio Connection Refused")
except Exception as e:
    logger.error(f"Minio make bucket error: {e}")


def no_wait(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, *kwargs)

    return wrapped


def get_image_from_storage_service(bucket: str, file_name: str) -> np.array:
    """
    get image from storage service (minio)
    @param bucket: bucket name
    @param file_name: file path
    @return:
    """
    try:
        image_object = minio_client.get_object(bucket, file_name)
        numpy_array = np.fromstring(image_object.read(), np.uint8)
        return cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
    except MaxRetryError:
        logger.error("Minio Connection Refused")
        get_minio_client()
    except Exception as ex:
        logger.error(f"Minio get object error: {e}")
    finally:
        image_object.close()
        image_object.release_conn()


def _upload_image(bucket_name: str, buffer: bytes, file_path: str, metadata=None):
    try:
        minio_client.put_object(bucket_name=bucket_name,
                                object_name=file_path,
                                data=io.BytesIO(buffer),
                                length=len(buffer), content_type="image/jpeg",
                                metadata=metadata)
    except MaxRetryError:
        logger.error("Minio Connection Refused")
        get_minio_client()
    except Exception as ex:
        logger.error(f"Minio Upload image error: {ex}")


def upload_image(
        bucket_name: str,
        image: np.ndarray,
        image_path: str,
        metadata=None
):
    try:
        buffer = cv2.imencode(".jpg", image)[1].tobytes()

        minio_client.put_object(bucket_name=bucket_name,
                                object_name=image_path,
                                data=io.BytesIO(buffer),
                                length=len(buffer), content_type="image/jpeg",
                                metadata=metadata)
    except MaxRetryError:
        logger.error("Minio Connection Refused")
        get_minio_client()
    except Exception as ex:
        logger.error(f"Minio Upload image error: {ex}")
    return os.path.join(bucket_name, image_path)




def upload_video(video_path, object_name):
    is_remove = False
    try:
        minio_client.fput_object(
            bucket_name=settings.minio_video_bucket_name, object_name=object_name, file_path=video_path,
            content_type="audio/mp4"
        )
        is_remove = True
        logger.debug(f"Upload video to Minio successful | video_path: {video_path} | object_name: {object_name}")
    except Exception as ex:
        logger.error(f"Upload video to Minio Failed | video_path: {video_path} | object_name: {object_name} | ex: {ex}")

    if is_remove:
        # remove file video in local :(.
        try:
            os.remove(video_path)
        except Exception as ex:
            logger.error(f"Can't remove video local {ex.__class__}, video_path: {video_path}")
