from typing import Tuple, List

import numpy as np
from log import logger
from service.inheritance.triton_serving_service import TritonServingService

from core.face_utils.box_utils import sort_by_size
from core.face_utils.face_processing import align_face


class CoreService:
    def __init__(self):
        self.serving_service = TritonServingService()

   
    def detect_faces(self, image_matrix, detection_threshold: float, limit=None) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect faces (mask) from numpy image
        Args:
            image_matrix: np.ndarray image of shape H x W x 3
            limit: maximum number of faces to return
        Returns:
            bounding_boxes: (np.ndarray) of int in format XYXY with shape (N x 4)
            landmarks: (np.ndarray) of int in format (XY XY XY XY XY) with shape (N x 10)
            confidences: (np.ndarray) of float with shape (N, 1)
        """

        boxes, landmarks, scores = self.serving_service.detect_faces(image_matrix, detection_threshold)
        if len(boxes) == 0:
            # Return placeholder for boxes, landmarks, and scores
            # NOT different for mask or non-mask
            return np.empty((0, 4)), np.empty((0, 10)), np.empty((0, 1))

        return sort_by_size(boxes, landmarks, scores, limit=limit)

    @classmethod
    def align_faces(cls, image_matrix, landmarks: np.ndarray) -> list:
        """
        Align faces from image and list of landmarks within that image
        Args:
            image_matrix: np.ndarray image of shape H x W x 3
            landmarks: np.ndarray of int in XY format with shape (N x 10)
            face_verifier: FaceVerification instance, default to None which means no verification
        Returns:
            List of aligned faces
        """
        assert len(landmarks.shape) == 2 and landmarks.shape[1] == 10, \
            logger.error("Expect landmarks with shape (N x 10), got: {}".format(landmarks.shape))

        aligned_faces = []
        for landmark in landmarks:
            aligned_face = align_face(image_matrix=image_matrix, landmark=landmark)

            aligned_faces.append(aligned_face)

        return aligned_faces

    def extract_feature_vectors(self, aligned_faces: List[np.ndarray]) -> np.ndarray:
        """
        Extract feature vectors from list of aligned face
        Args:
            aligned_faces: List of aligned_face
        Returns:
            feature_vectors: np.ndarray of feature vectors of float32 with shape (N x 512)
        """
        if len(aligned_faces) == 0:
            return np.empty((0, 512))
        return self.serving_service.extract_features(np.array(aligned_faces))


