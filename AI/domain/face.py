import uuid
from typing import List, Optional

import numpy as np
# from iva.util.entity import Area, Point

from core.config import get_app_settings
from domain.common import DateTimeModelMixin
from domain.rwmodel import RWModel


class FaceAttribute:
    attribute_name: str
    attribute_score: int
    attribute_confidence: float

    def __init__(self, attribute_name=None, attribute_score=None, attribute_confidence=1.0):
        self.attribute_name = attribute_name
        self.attribute_score = attribute_score
        self.attribute_confidence = attribute_confidence


class FaceInfo:
    track_id: Optional[int]
    track_init_time: Optional[int]
    bounding_box: np.ndarray  # x1, y1, x2, y2
    detection_score: float
    landmarks: np.ndarray
    feature_vector: Optional[np.ndarray]
    face_attributes: Optional[List[FaceAttribute]]
    face_image_url: Optional[str]
    matched_face_token: Optional[str]
    history_face_token: Optional[str]
    topK_similar: Optional[list]
    recognize_confidence: float
    person_id: Optional[str]
    video_url: Optional[str]

    def __init__(self, bounding_box: np.ndarray, detection_score: float, landmarks: np.ndarray):
        self.bounding_box = bounding_box
        self.detection_score = detection_score
        self.landmarks = landmarks
        self.track_id = None
        self.feature_vector = np.empty(shape=(1, 512))
        self.face_attributes = []
        self.face_image_url = None
        self.matched_face_token = None
        self.history_face_token = None
        self.recognize_confidence = 0
        self.person_id = ""
        self.video_url = None
        self.track_init_time = 0
        self.topK_similar = None


class Face(DateTimeModelMixin, RWModel):
    face_token: Optional[str]
    person_id: Optional[str]
    detection_score: Optional[float]
    bounding_box: Optional[dict]
    landmarks: Optional[List[dict]]
    face_attributes: Optional[List[dict]]
    image_url: Optional[str]
    feature_vector: Optional[List[float]]

    def __init__(self, face_info: FaceInfo, person_id: str, width: int, height: int):
        super().__init__()
        self.face_token = str(uuid.uuid4().int & (1 << 63) - 1)
        self.person_id = person_id
        self.detection_score = float(face_info.detection_score)
        self.bounding_box = {
            "x": float(face_info.bounding_box[0] / width),
            "y": float(face_info.bounding_box[1] / height),
            "width": float((face_info.bounding_box[2] - face_info.bounding_box[0]) / width),
            "height": float((face_info.bounding_box[3] - face_info.bounding_box[1]) / height)}
        self.landmarks = [
            {"x": float(face_info.landmarks[0] / width), "y": float(face_info.landmarks[1] / height)},
            {"x": float(face_info.landmarks[2] / width), "y": float(face_info.landmarks[3] / height)},
            {"x": float(face_info.landmarks[4] / width), "y": float(face_info.landmarks[5] / height)},
            {"x": float(face_info.landmarks[6] / width), "y": float(face_info.landmarks[7] / height)},
            {"x": float(face_info.landmarks[8] / width), "y": float(face_info.landmarks[9] / height)}
        ]
        self.face_attributes = [{
            "name": face_attribute.attribute_name,
            "confidence": float(face_attribute.attribute_confidence * 100),
            "value": face_attribute.attribute_score
        }
            for face_attribute in face_info.face_attributes]
        self.image_url = face_info.face_image_url
        self.feature_vector = face_info.feature_vector.tolist()

    def to_dict(self):
        return {
            "face_token": self.face_token,
            "person_id": self.person_id,
            "detection_score": self.detection_score,
            "bounding_box": self.bounding_box,
            "landmarks": self.landmarks,
            "face_attributes": self.face_attributes,
            "image_url": self.image_url,
        }

    def to_camel_dict(self):
        return {
            "faceToken": self.face_token,
            "detectScore": self.detection_score,
            "boundingBox": self.bounding_box,
            "landmarks": {"points": self.landmarks},
            "faceAttributes": self.face_attributes,
            "imageUrl": self.image_url,
        }


class CompareFaceResponse:
    faces_metadata_1: Face
    faces_metadata_2: Face
    similarity_confidence: float
    is_same_person: bool

    def to_dict(self):
        return {
            "faces_metadata_1": self.faces_metadata_1.to_dict(),
            "faces_metadata_2": self.faces_metadata_2.to_dict(),
            "similarityConfidence": self.similarity_confidence,
            "isSamePerson": self.is_same_person
        }
    
    def to_camel_dict(self):
        return {
            "facesMetadata1": self.faces_metadata_1.to_camel_dict(),
            "facesMetadata2": self.faces_metadata_2.to_camel_dict(),
            "similarityConfidence": self.similarity_confidence,
            "samePerson": self.is_same_person
        }




