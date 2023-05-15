import os
from datetime import datetime
from typing import List

import cv2
import numpy as np
import psycopg2
from celery.signals import worker_process_init
from log import logger
from service.face_management import FaceManagementService
from storage_service import get_image_from_storage_service
from milvus.client.exceptions import ConnectError
from psycopg2 import pool

from core.config import get_app_settings
from db.repository.face_repository import save_face_tokens
from domain.common import CommonResponse, Status
from domain.face import Face, CompareFaceResponse

model_instance = None
face_management = FaceManagementService()
db_pool = psycopg2.pool.ThreadedConnectionPool(
    settings.min_connection_count,
    settings.max_connection_count,
    "postgresql://postgres:changeme@localhost:5432/face_management"
)
face_image = None


def register_face(bucket_name: str, image_paths: List[str], person_id: str, faceset_tokens: List[str], created_by: str,
                  enable_generate_mask_face: bool):
    responses = []
    try:
        # save to database
        db_connection = None
        db_connection = db_pool.getconn()
        faces = []
        for image_path in image_paths:
            image_name = os.path.basename(image_path)[9:]
            image = get_image_from_storage_service(bucket=bucket_name, file_name=image_path)
            crop_base_image_path = os.path.splitext(image_path.replace("raw", "cropped"))[0]
            logger.info(f"crop_base_image_path: {crop_base_image_path}")
            result: CommonResponse = face_image.register_face(
                image,
                upload_image=True,
                base_crop_image_path=crop_base_image_path,
                enable_gen_mask=enable_generate_mask_face
            )
            result.image_name = image_name
            if result.status is not Status.SUCCESS.value:
                responses.append(result.to_dict())
            else:
                height, width, _ = image.shape
                # result_dicts = []
                for detected_face in result.data:
                    face = Face(face_info=detected_face, person_id=person_id, width=width, height=height)
                    faces.append(face)
                    result_dict = face.to_dict()
                    result_dict["image_name"] = image_name
                    responses.append(result_dict)
                    # result_dicts.append(result_dict)
                # responses.append(result_dicts)

        if len(faces) > 0:
            success = save_face_tokens(
                connection=db_connection,
                faces=faces,
                faceset_tokens=faceset_tokens,
                created_by=created_by
            )
            if not success:
                logger.info("save face token failed")
                return CommonResponse(status=Status.FAILED,
                                      code="ERROR_916",
                                      message="Database Error").to_dict()

            for faceset_token in faceset_tokens:
                face_tokens = [face.face_token for face in faces]
                logger.info("Add face_tokens: {} to faceset: {}".format(face_tokens, faceset_token))
                face_management.add_face_tokens_to_faceset(
                    face_tokens=face_tokens,
                    feature_vectors=[face.feature_vector for face in faces],
                    faceset_token=faceset_token
                )
    except Exception as e:
        logger.info(e)
        return CommonResponse(status=Status.FAILED, message=e.__class__.__name__, code="400").to_dict()
        # raise e
    finally:
        if db_connection is not None:
            db_pool.putconn(db_connection)
    return responses


def add_face_token_to_faceset(face_token: str, feature_vector: List[float], faceset_token: str) -> None:
    logger.debug("Add face_token={} to faceset={}".format(face_token, faceset_token))
    face_management.add_face_token_to_faceset(face_token, feature_vector, faceset_token)
    return None
