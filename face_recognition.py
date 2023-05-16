import cv2
import psycopg2

from core.config import get_app_settings
from core.face_utils.visualize import draw_detected_faces
from core.tracker.sort import Sort

from service.face_image_service import FaceImageService

import time

settings = get_app_settings()

face_image_service = FaceImageService()
recognition_threshold = 75
detection_threshold = 60
faceset_token = "_c3770594291f452782520613dfeaa6c"
tracker = Sort(max_age=3, min_hits=0, max_distance=0.82,
               recognition_threshold=0.75)

db_pool = psycopg2.pool.ThreadedConnectionPool(
    settings.min_connection_count,
    settings.max_connection_count,
    "postgresql://postgres:changeme@localhost:5432/face_management"
)
db_connection = db_pool.getconn()


def recognition(frame):
    timestamp = int(time.time())
    detected_faces, log_message = face_image_service.detect_faces(
        image=frame,
        detection_threshold=detection_threshold,
    )
    frame_height, frame_width, _ = frame.shape

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

    person_names = []
    for face_info in detected_faces:
        if face_info.recognize_confidence >= recognition_threshold:
            print(face_info.person_id)
            person_names.append(face_info.person_id)

    return person_names

# frame = cv2.imread("daniels.1.1.png")
# recognition(frame)
