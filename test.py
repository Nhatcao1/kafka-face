import cv2
import psycopg2


from core.config import get_app_settings
from core.face_utils.visualize import draw_detected_faces

from core.tracker.sort import Sort

from domain.face import FaceInfo
from service.face_image_service import FaceImageService

import time

settings = get_app_settings()


# cap = cv2.VideoCapture("videos/daniels.mp4")
cap = cv2.VideoCapture("test_dat.mp4")

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

    frame = draw_detected_faces(frame, detected_faces, recognition_threshold)

    frame = cv2.resize(frame, (1280, 720), cv2.INTER_AREA)
    cv2.imshow("output", frame)
    cv2.waitKey(1)
    # out.write(frame)
# out.release()
cap.release()
