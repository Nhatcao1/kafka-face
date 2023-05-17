import datetime

import cv2
import psycopg2

import storage_service
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




sent_event_recognized_track_ids = []
sent_event_unknown_track_ids = []

frame = cv2.imread("dataset/Kita/Kita.4.1.png")
timestamp = int(time.time())
detected_faces, log_message = face_image_service.detect_faces(
    image=frame,
    detection_threshold=detection_threshold,
    upload_cropped_image=False,
    base_image_path="test"
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


frame = draw_detected_faces(frame, detected_faces, recognition_threshold,)
timestamp = int(time.time())

timestamp = datetime.datetime.fromtimestamp(timestamp)


for face_info in detected_faces:
    if face_info.recognize_confidence >= recognition_threshold:
        x1,y1, x2, y2 = face_info.bounding_box
        cropped_face = frame[y1:y2, x1:x2]
        if face_info.person_id is None or len(face_info.person_id) == 0:
            person_name = ""
        else:
            person_name = face_info.person_id
        try:
            img_name = "{}-{}-{}/{}_{}.jpg".format(
                timestamp.year,
                timestamp.month,
                timestamp.day,
                datetime.datetime.today().strftime("%H:%M:%S"), person_name)
            image_url = storage_service.upload_image(
                bucket_name=settings.minio_event_bucket_name,
                image=cropped_face,
                image_path=img_name
            )
            print(image_url)
        except Exception:
            print(Exception)
        print(face_info.person_id)

# frame = cv2.resize(frame)
cv2.imshow("output", frame)
# cv2.imwrite("Dat.png", frame)
cv2.waitKey(0)
    # out.write(frame)

