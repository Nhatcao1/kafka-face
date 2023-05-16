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


def detect_faces(image: np.array, upload_cropped_image=True, base_crop_image_path: str = str(uuid.uuid4())) -> (
List[FaceInfo], str):
    print(f"base crop image path: {base_crop_image_path}")
    face_image_service = FaceImageService()
    detected_faces, log_message = face_image_service.detect_faces(
        image=image,
        upload_cropped_image=upload_cropped_image,
        base_image_path=base_crop_image_path)
    if len(detected_faces) < 1:
        # print("No face found")
        print(log_message)
        print("----------------------------------------------")
        return [], log_message

    # STEP 2: get feature vectors from detection result
    face_image_service.extract_embeddings_from_detections(image=image, faces_info=detected_faces)
    return detected_faces, log_message


cap = cv2.VideoCapture("daniels.mp4")
# out = cv2.VideoWriter('cam_8_toi.avi',
#                       cv2.VideoWriter_fourcc(*'MJPG'),
#                       35, (1280, 720))
db_pool = psycopg2.pool.ThreadedConnectionPool(
    settings.min_connection_count,
    settings.max_connection_count,
    "postgresql://postgres:changeme@localhost:5432/face_management"
)
db_connection = db_pool.getconn()
face_image_service = FaceImageService()
recognition_threshold = 75
detection_threshold = 60
faceset_token = "_c3770594291f452782520613dfeaa6c"
tracker = Sort(max_age=3, min_hits=0, max_distance=0.82,
               recognition_threshold=0.75)

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
        base_image_path="test"
    )
    frame_height, frame_width, _ = frame.shape

    print(len(detected_faces))

    face_image_service.recognition_detected_faces(
        frame=frame,
        timestamp=timestamp,
        detected_faces=detected_faces,
        recognition_threshold=recognition_threshold,
        faceset_token=faceset_token,
        tracker=tracker,
        db_connection=db_connection
    )

    frame = draw_detected_faces(frame, detected_faces, recognition_threshold,
                                timestamp)

    frame = cv2.resize(frame, (1280, 720), cv2.INTER_AREA)
    cv2.imshow("output", frame)
    cv2.waitKey(1)
    # out.write(frame)
# out.release()
cap.release()
