import cv2
import json
import numpy as np
import psycopg2
import uuid
from log import logger

from typing import List

from core.config import get_app_settings
from core.face_utils.visualize import draw_detected_faces

from core.tracker.sort import Sort

from domain.face import FaceInfo
from service.face_image_service import FaceImageService

import time

settings = get_app_settings()


class Model:
    feature_name = "Facial Analysis"
    description = ""
    config_format = json.load(open("resource/config_format.json"))


def detect_faces(image: np.array, upload_cropped_image=False, base_crop_image_path: str = str(uuid.uuid4())) \
        -> (List[FaceInfo], str):
    face_image_service = FaceImageService()
    detected_faces, log_message = face_image_service.detect_faces(
        image=image,
        upload_cropped_image=upload_cropped_image,
        base_image_path=base_crop_image_path,
        verify=True)
    if len(detected_faces) < 1:
        # logger.info("No face found")
        logger.info(log_message)
        logger.info("----------------------------------------------")
        return [], log_message

    # STEP 2: get feature vectors from detection result
    face_image_service.extract_embeddings_from_detections(image=image, faces_info=detected_faces)
    return detected_faces, log_message


def analyze_video():
    cap = cv2.VideoCapture("test.mp4")
    out = cv2.VideoWriter('cam_8_toi.avi',
                          cv2.VideoWriter_fourcc(*'MJPG'),
                          35, (1280, 720))
    # db_pool = psycopg2.pool.ThreadedConnectionPool(
    #     settings.min_connection_count,
    #     settings.max_connection_count,
    #     str(settings.database_url)
    # )
    face_image_service = FaceImageService()
    recognition_threshold = 0.75
    detection_threshold = 0.6
    faceset_token = "_c211a43996114ae0b5f39ba3eba973f"
    tracker = Sort(max_age=3, min_hits=0, max_distance=0.82,
                   recognition_threshold=recognition_threshold)

    sent_event_recognized_track_ids = []
    sent_event_unknown_track_ids = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = int(time.time())
        detected_faces, log_message = face_image_service.detect_faces(
            image=frame,
            detection_threshold=detection_threshold,
            upload_cropped_image=False,
            bucket_name="iva-bucket"
        )
        frame_height, frame_width, _ = frame.shape

        logger.info(len(detected_faces))
        # db_connection = db_pool.getconn()
        face_image_service.recognition_detected_faces(
            frame=frame,
            timestamp=timestamp,
            detected_faces=detected_faces,
            recognition_threshold=recognition_threshold,
            faceset_token=faceset_token,
            tracker=tracker,
            db_connection=db_connection
        )
        db_pool.putconn(db_connection)

        frame = draw_detected_faces(frame, detected_faces, recognition_threshold,
                                    timestamp)

        recognized_faces = []
        recognized_face_name = ""
        for face in detected_faces:
            if face.recognize_confidence >= recognition_threshold:
                recognized_face_name += face.person_id + ", "
                recognized_faces.append(face)

        recognized_face_name = recognized_face_name[:-2] + "."
        recognized_face_event = [face for face in recognized_faces if face.track_id not in
                                 sent_event_recognized_track_ids]

        recognized_track_ids = [face.track_id for face in recognized_face_event]
        enable_send_unknown_event = False if len(recognized_face_event) > 0 else True

        if len(recognized_face_event) > 0:
            event_in_frame = True
            sent_event_recognized_track_ids.extend(recognized_track_ids)
            del sent_event_recognized_track_ids[0: max(0, (len(sent_event_recognized_track_ids) - 100))]

        unknown_faces = [
            face
            for face in detected_faces
            if face.recognize_confidence < recognition_threshold
        ]
        unknown_face_track_ids = [face.track_id for face in unknown_faces if (
                1000 < timestamp - face.track_init_time and face.track_id not in sent_event_unknown_track_ids)]
        if len(unknown_face_track_ids) > 0:
            event_in_frame = True
            sent_event_unknown_track_ids.extend(unknown_face_track_ids)
            del sent_event_unknown_track_ids[0: max(0, (len(sent_event_unknown_track_ids) - 100))]
        frame = cv2.resize(frame, (1280, 720), cv2.INTER_AREA)
        out.write(frame)
    out.release()
    cap.release()
