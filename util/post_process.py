from typing import List
import numpy as np
import datetime
from iva.util.entity import Label, Instance, BoundingBox
from domain.face import FaceInfo
from iva.service import storage_service
from core.face_utils.box_utils import expand_coordinates


def get_labels_from_face_infos(image: np.array, face_infos: List[FaceInfo], upload_cropped_image, config_id) -> List[Label]:
    height, width, _ = image.shape
    instances: List[Instance] = []
    base_image_path = "{}/Facial_Analysis/{}".format(datetime.datetime.today().strftime('%Y-%m-%d'), config_id)
    for i, face_info in enumerate(face_infos):
        image_url = ""
        if upload_cropped_image:
            img_name = "{}/{}_{}.jpg".format(base_image_path, datetime.datetime.today().strftime("%H:%M:%S"), i + 1)
            x1, y1, x2, y2 = expand_coordinates(frame=image, bbox=face_info.bounding_box, expand_ratio=1.5)
            cropped_face = image[y1:y2, x1:x2]
            try:
                image_url = storage_service.upload_image(
                    bucket_name="iva-bucket",
                    image=cropped_face,
                    image_path=img_name
                )
            except Exception:
                pass
        instances.append(
            Instance(
                bounding_box=BoundingBox(
                    x1=face_info.bounding_box[0] / width,
                    y1=face_info.bounding_box[1] / height,
                    x2=face_info.bounding_box[2] / width,
                    y2=face_info.bounding_box[3] / height
                ),
                confidence=face_info.detection_score * 100,
                additional_info={
                    "track_id": face_info.track_id,
                    "person_id": face_info.person_id,
                    "matched_face_token": str(face_info.matched_face_token),
                    "history_face_token": str(face_info.history_face_token),
                    "recognize_confidence": face_info.recognize_confidence,
                    "landmarks": [
                        {"x": float(face_info.landmarks[0] / width), "y": float(face_info.landmarks[1] / height)},
                        {"x": float(face_info.landmarks[2] / width), "y": float(face_info.landmarks[3] / height)},
                        {"x": float(face_info.landmarks[4] / width), "y": float(face_info.landmarks[5] / height)},
                        {"x": float(face_info.landmarks[6] / width), "y": float(face_info.landmarks[7] / height)},
                        {"x": float(face_info.landmarks[8] / width), "y": float(face_info.landmarks[9] / height)}
                    ],
                    "image_url": image_url,
                    "face_attributes": [{
                        "attribute_name": face_attribute.attribute_name,
                        "attribute_score": face_attribute.attribute_score
                    }
                        for face_attribute in face_info.face_attributes]
                }
            )
        )
    return [Label(name="face", instances=instances)]
