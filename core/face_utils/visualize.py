from typing import List

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from domain.face import FaceInfo

import time
import datetime
import unidecode

LEFT_EYE = (102, 140, 255)
RIGHT_EYE = (102, 179, 255)
NOSE = (102, 255, 140)
LEFT_MOUTH = (255, 179, 102)
RIGHT_MOUTH = (255, 102, 179)

LIST_COLORS = [LEFT_EYE, RIGHT_EYE, NOSE, LEFT_MOUTH, RIGHT_MOUTH]


def box_xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def visualize_landmark(image: np.ndarray, landmark: np.ndarray):
    # Except [xy xy xy xy xy] shape landmark
    assert len(landmark) == 10, "Expect landmark with shape (10, ), got {}".format(landmark.shape)

    landmark = landmark.reshape((-1, 2))
    for (x, y), color in zip(landmark, LIST_COLORS):
        cv2.circle(image, (x, y), radius=2, color=color, thickness=-1)


def draw_detected_faces(frame: np.ndarray, detected_faces: List[FaceInfo], recognize_threshold: float, timestamp: int):
    frame_height, frame_width, _ = frame.shape
    for face_info in detected_faces:
        box_color = (0, 0, 255)
        text = "{}|unknown|{}|{}".format(face_info.track_id, "{:.2f}".format(face_info.detection_score),
                                         "{:.2f}".format(face_info.recognize_confidence))
        if face_info.recognize_confidence >= recognize_threshold:
            box_color = (0, 255, 0)
            if face_info.person_id is None:
                text = "{}|{}|{}|{}".format(face_info.track_id, face_info.matched_face_token,
                                            "{:.2f}".format(face_info.detection_score),
                                            "{:.2f}".format(face_info.recognize_confidence))
            else:
                text = "{}|{}|{}|{}".format(face_info.track_id, unidecode.unidecode(face_info.person_id),
                                            "{:.2f}".format(face_info.detection_score),
                                            "{:.2f}".format(face_info.recognize_confidence))

        rect_width = (face_info.bounding_box[2] - face_info.bounding_box[0]) / 6
        rect_height = (face_info.bounding_box[3] - face_info.bounding_box[1]) / 6
        # top left
        cv2.polylines(frame, [np.array([[face_info.bounding_box[0], face_info.bounding_box[1] + rect_height],
                                        [face_info.bounding_box[0], face_info.bounding_box[1]],
                                        [face_info.bounding_box[0] + rect_width, face_info.bounding_box[1]]],
                                       np.int32).reshape((-1, 1, 2))], False, box_color, 2)

        # top right
        cv2.polylines(frame, [np.array([[face_info.bounding_box[2] - rect_width, face_info.bounding_box[1]],
                                        [face_info.bounding_box[2], face_info.bounding_box[1]],
                                        [face_info.bounding_box[2], face_info.bounding_box[1] + rect_height]],
                                       np.int32).reshape((-1, 1, 2))], False, box_color, 2)

        # bottom right
        cv2.polylines(frame, [np.array([[face_info.bounding_box[2] - rect_width, face_info.bounding_box[3]],
                                        [face_info.bounding_box[2], face_info.bounding_box[3]],
                                        [face_info.bounding_box[2], face_info.bounding_box[3] - rect_height]],
                                       np.int32).reshape((-1, 1, 2))], False, box_color, 2)

        # bottom left
        cv2.polylines(frame, [np.array([[face_info.bounding_box[0], face_info.bounding_box[3] - rect_height],
                                        [face_info.bounding_box[0], face_info.bounding_box[3]],
                                        [face_info.bounding_box[0] + rect_width, face_info.bounding_box[3]]],
                                       np.int32).reshape((-1, 1, 2))], False, box_color, 2)

        cv2.rectangle(frame, (int(face_info.bounding_box[0]), int(face_info.bounding_box[1])),
                      (int(face_info.bounding_box[2]), int(face_info.bounding_box[1] - frame_height * 0.02)),
                      box_color, -1)

        cv2.putText(frame, text, (int(face_info.bounding_box[0]), int(face_info.bounding_box[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return frame
