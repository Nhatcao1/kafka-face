import copy
import time
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import cv2
import numpy as np
from log import logger
import storage_service
from psycopg2.pool import AbstractConnectionPool

from core.config import get_app_settings
from core.face_utils import face_processing
from core.face_utils.box_utils import expand_coordinates, expand_coordinates_1


from domain.common import CommonResponse, Status
from domain.face import FaceInfo
from service import tracking_service
from service.core_service import CoreService
from service.face_management import FaceManagementService

settings = get_app_settings()


def get_dates_from_date_range(
        start_date: datetime,
        end_date: datetime
) -> List[str]:
    history_faceset_tokens = []
    delta_date = end_date - start_date
    for i in range(delta_date.days + 1):
        current_date = start_date + timedelta(days=i)
        logger.info(
            f"start_date: {start_date} | end_date: {end_date} | delta_date: {delta_date.days + 1} | current_date = {current_date}")
        history_faceset_token = f"{current_date.year}_{current_date.month}_{current_date.day}"
        history_faceset_tokens.append(history_faceset_token)
    if f"{end_date.year}_{end_date.month}_{end_date.day}" not in history_faceset_tokens:
        history_faceset_tokens.append(f"{end_date.year}_{end_date.month}_{end_date.day}")
    return history_faceset_tokens


class FaceImageService:
    def __init__(self):
        self.core_service = CoreService()
        self.face_management_service = FaceManagementService()

        self.face_registration_bucket_name = settings.face_registration_bucket_name

    def detect_faces(
            self,
            image: np.ndarray,
            detection_threshold: float = 60,
            upload_cropped_image: bool = False,
            base_image_path: str = "",
            bucket_name: str = settings.face_registration_bucket_name,
    ) -> Tuple[List[FaceInfo], str]:
        face_detection_start_time = time.time()
        boxes, landmarks, scores = self.core_service.detect_faces(
            image_matrix=image,
            detection_threshold=detection_threshold)

        print("{} faces detected in time: {}s".format(
            len(boxes), round(time.time() - face_detection_start_time, 4)))
        detected_faces = [FaceInfo(
            bounding_box=box,
            detection_score=float(score[0]),
            landmarks=landmark
        )
            for box, score, landmark in zip(boxes, scores, landmarks)
        ]
        # validate faces
        for i, detected_face in enumerate(detected_faces):

            bounding_box = detected_face.bounding_box
            image_height, image_width, _ = image.shape
            im_h = bounding_box[3]
            im_w = bounding_box[2]
            if not (80 < im_w) or \
                    not (80 < im_h):
                detected_faces.remove(detected_face)
                continue

            if upload_cropped_image:
                # upload to minio -> update face_image_url
                img_name = "{}/{}_{}.jpg".format(base_image_path, datetime.today().strftime("%H:%M:%S"), i + 1)
                bbox = detected_face.bounding_box
                x1, y1, x2, y2 = bbox
                # x1, y1, x2, y2 = expand_coordinates_1(frame=image, bbox=bbox, expand_ratio=2.5)
                cropped_face = image[y1:y2, x1:x2]
                print(f"bucket name: {bucket_name}, image_path: {img_name}")
                # try:
                image_url = storage_service.upload_image(
                    bucket_name=bucket_name,
                    image=cropped_face,
                    image_path=img_name
                )
                detected_face.face_image_url = image_url
                # except Exception:
                #
                #     pass
        log_message = f"found {len(boxes)} face (s) "
        return detected_faces, log_message

    def extract_embeddings_from_detections(
            self,
            image: np.ndarray,
            faces_info: List[FaceInfo]
    ) -> None:
        align_face_start_time = time.time()
        aligned_faces = [face_processing.align_face(image_matrix=image, landmark=np.array(face_info.landmarks))
                         for face_info in faces_info]
        logger.debug("Align {} face(s) time: {}s| ".format(
            len(aligned_faces),
            round(time.time() - align_face_start_time, 4)))
        # STEP 4: extract features
        extract_feature_start_time = time.time()
        feature_vectors = self.core_service.extract_feature_vectors(aligned_faces=aligned_faces)
        logger.debug("Extract {} feature(s) time: {}s| ".format(
            len(feature_vectors),
            round(time.time() - extract_feature_start_time, 4)
        ))
        for detected_face, feature_vector in zip(faces_info, feature_vectors):
            detected_face.feature_vector = feature_vector

    def recognition_detected_faces(
            self,
            frame: np.ndarray,
            timestamp: int,
            detected_faces: List[FaceInfo],
            faceset_token: str,
            recognition_threshold: float,
            tracker,
            db_connection: AbstractConnectionPool
    ) -> None:
        if len(detected_faces) < 1:
            logger.debug("No face found!")
            logger.debug("----------------------------------------------")
            if tracker is not None:
                tracker.update([])
                return

        if tracker is not None:
            start_time = time.time()
            self.extract_embeddings_from_detections(frame, detected_faces)
            active_tracks = tracker.update(detected_faces, timestamp)
            tracking_service.get_info_from_track(detected_faces, active_tracks)

            unrecognized_faces = [detected_face
                                  for detected_face in detected_faces
                                  if detected_face.recognize_confidence < recognition_threshold]
        else:
            unrecognized_faces = detected_faces

        if len(unrecognized_faces) != 0:
            self.face_management_service.find_face_token_from_feature_vectors(
                face_info_list=unrecognized_faces,
                faceset_token=faceset_token,
                recognition_threshold=recognition_threshold,
                db_connection=db_connection
            )
            if tracker is not None:
                tracking_service.update_tracks_info(unrecognized_faces, tracker)

    def get_face_tokens_by_image(
            self,
            face_info: FaceInfo,
            image: np.ndarray,
            top_k: int,
            faceset_token: str,
            partition_tags: Optional[List[str]] = None
    ):
        largest_face: List[FaceInfo] = [face_info]
        self.extract_embeddings_from_detections(
            image=image,
            faces_info=largest_face
        )
        face_tokens, similarity_scores = self.face_management_service.find_face_tokens_from_faceset(
            faceset_token=faceset_token,
            partition_tags=partition_tags,
            top_k=top_k,
            feature_vector=largest_face[0].feature_vector.tolist()
        )
        return face_tokens, similarity_scores.tolist()

    def register_face(
            self,
            image,
            upload_image,
            base_crop_image_path,
            enable_gen_mask
    ) -> CommonResponse:

        register_faces = []
        register_images = [image]
        aligned_faces = []
        detected_faces, log_message = self.detect_faces(
            image=image,
            upload_cropped_image=True,
            base_image_path=base_crop_image_path,
            verify=False
        )
        if len(detected_faces) < 1:
            return CommonResponse(status=Status.BAD_REQUEST, message="No face found", code="ERROR_914")
        elif len(detected_faces) > 1:
            return CommonResponse(status=Status.BAD_REQUEST, message="Number of faces greater than 1", code="ERROR_915")

        largest_face: FaceInfo = detected_faces[0]
        x1, y1, x2, y2 = expand_coordinates(frame=image, bbox=largest_face.bounding_box, expand_ratio=1.25)
        if upload_image:
            img_name = "{}.jpg".format(base_crop_image_path)
            cropped_face = image[y1:y2, x1:x2]
            image_url = storage_service.upload_image(
                bucket_name=self.face_registration_bucket_name,
                image=cropped_face,
                image_path=img_name
            )
            logger.info(f"image_url: {image_url}")
            logger.info(f"img_name: {img_name}")
            largest_face.face_image_url = image_url

        register_faces.append(largest_face)
        if enable_gen_mask:
            mask_images = self.mask_face_service.get_mask_face(image=image, face_location=largest_face.bounding_box)
            for i, mask_image in enumerate(mask_images):
                register_images.append(mask_image)
                mask_face = copy.deepcopy(largest_face)
                if upload_image:
                    # upload to minio -> update face_image_url
                    img_name = "{}_mask_{}.jpg".format(base_crop_image_path, i + 1)
                    cropped_face = mask_image[y1:y2, x1:x2]
                    image_url = storage_service.upload_image(
                        bucket_name=self.face_registration_bucket_name,
                        image=cropped_face,
                        image_path=img_name
                    )
                    mask_face.face_image_url = image_url
                register_faces.append(mask_face)
        for register_face, register_image in zip(register_faces, register_images):
            aligned_faces.append(
                face_processing.align_face(image_matrix=register_image, landmark=np.array(register_face.landmarks)))

        feature_vectors = self.core_service.extract_feature_vectors(aligned_faces=aligned_faces)
        for register_face, feature_vector in zip(register_faces, feature_vectors):
            register_face.feature_vector = feature_vector
        return CommonResponse(data=register_faces)

    def register_face_by_event(self, frame: np.ndarray):
        frame = cv2.resize(frame, (112, 112))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        feature_vectors = self.core_service.extract_feature_vectors(aligned_faces=[frame])
        detected_face = FaceInfo(
            bounding_box=np.array([0] * 4),
            detection_score=100,
            landmarks=np.array([0] * 10)
        )
        detected_face.feature_vector = feature_vectors[0]
        return [detected_face]
