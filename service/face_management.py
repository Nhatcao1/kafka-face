from typing import List, Optional, Tuple

import numpy as np
from log import logger
from psycopg2.pool import AbstractConnectionPool

from core.config import get_app_settings
from db.repository.face_repository import get_person_id_from_face_token, get_image_url_from_face_token
from domain.face import FaceInfo
from error import error_code
from error.customized_exception import InternalException
from service.inheritance.milvus_service import MilvusService


class FaceManagementService:
    def __init__(self):
        self.settings = get_app_settings()
        self.feature_vector_service = MilvusService()
        self.last_retry = 0

    def get_person_id_from_face_token(self, face_token: str, db_connection: AbstractConnectionPool) -> str:
        return get_person_id_from_face_token(connection=db_connection, face_token=face_token)

    def get_image_url_from_face_token(self, face_token: str, db_connection: AbstractConnectionPool) -> str:
        return get_image_url_from_face_token(connection=db_connection, face_token=face_token)

    def find_face_token_from_feature_vectors(
            self,
            face_info_list: List[FaceInfo],
            faceset_token: str,
            recognition_threshold: float,
            db_connection: AbstractConnectionPool,
    ) -> None:
        feature_vectors = np.array([face_info.feature_vector for face_info in face_info_list]).astype(np.float32)

        self.feature_vector_service.create_collection(
            collection_name=faceset_token
        )
        try:
            id_array, distance_array = self.feature_vector_service.search(
                query_vectors=feature_vectors,
                collection_name=faceset_token,
                top_k=3
            )
            id_array = np.array(id_array)
            vector_ids = id_array[:, 0]
            distance_array = np.array(distance_array)[:, 0].tolist()
        except TypeError:
            id_array, distance_array = None, []
        except IndexError:
            id_array, distance_array = None, []

        if id_array is None or len(distance_array) == 0:
            return
        distance_threshold = self.settings.distance_threshold
        if np.array(distance_array).shape in (0, 1):
            logger.error(
                "Bug code: expect distances with shape (N, ) or a scalar, but got: {}".format(distance_array.shape))
            raise InternalException(error_code.UNEXPECTED_ERROR)
        distances = np.sqrt(distance_array)
        distances[distances == 0] = distance_threshold
        similarity_scores = np.minimum(100, np.where(distances > 0, 100 * float(distance_threshold) / distances, 100))

        for face_info, vector_id, score, topK_vector in zip(face_info_list, vector_ids, similarity_scores, id_array):
            face_info.matched_face_token = vector_id
            face_info.topK_similar = topK_vector
            face_info.recognize_confidence = float(score)
            if face_info.recognize_confidence >= recognition_threshold and face_info.person_id == "":
                face_info.person_id = self.get_person_id_from_face_token(
                    face_token=str(face_info.matched_face_token),
                    db_connection=db_connection)

    def add_face_token_to_faceset(self, face_token: str, feature_vector: List[float], faceset_token: str,
                                  partition_tag: str = None):
        self.feature_vector_service.insert_vectors(
            vectors=[feature_vector],
            ids=[int(face_token)],
            collection_name=faceset_token,
            partition_tag=partition_tag
        )

    def add_face_tokens_to_faceset(self, face_tokens: List[str], feature_vectors: List[List[float]],
                                   faceset_token: str):
        # self.feature_vector_service.remove_collection(collection_name=faceset_token)
        self.feature_vector_service.insert_vectors(
            vectors=feature_vectors,
            ids=[int(face_token) for face_token in face_tokens],
            collection_name=faceset_token
        )

    def remove_face_token_in_faceset(self, face_token: str, faceset_token: str):
        self.feature_vector_service.remove_vectors(
            ids=[int(face_token)],
            collection_name=faceset_token
        )

    def find_face_tokens_from_faceset(
            self,
            faceset_token: str,
            partition_tags: Optional[List[str]],
            top_k: int,
            feature_vector: List[float]
    ) -> Tuple[List[str], np.ndarray]:
        self.feature_vector_service.create_collection(
            collection_name=faceset_token
        )
        ids, distance_arrays = self.feature_vector_service.search(
            query_vectors=np.array([feature_vector]).astype(np.float32),
            top_k=top_k,
            collection_name=faceset_token,
            partition_tags=partition_tags
        )
        if np.array(distance_arrays).shape in (0, 1):
            logger.error(
                "Bug code: expect distances with shape (N, ) or a scalar, but got: {}".format(distance_arrays.shape))
            raise InternalException(error_code.UNEXPECTED_ERROR)
        distances = np.sqrt(distance_arrays)
        distances[distances == 0] = self.settings.distance_threshold
        similarity_scores = np.minimum(100, np.where(distances > 0,
                                                     100 * float(self.settings.distance_threshold) / distances, 100))
        return ids, similarity_scores

    def is_faceset_token_exists(self, faceset_name: str):
        return self.feature_vector_service.is_collection_exists(collection_name=faceset_name)
