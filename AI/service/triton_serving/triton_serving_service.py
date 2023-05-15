from typing import List

import numpy as np
from numpy.linalg import norm

from service.triton_serving.face_detection_triton_inference import SCRFD
from service.triton_serving.face_recognition_triton_inference import FaceRecognitionTriton


class TritonServingService:
    def __init__(self):
        super().__init__()
        self.face_detector = SCRFD()
        self.face_detector.prepare(-1)
        self.face_recognition = FaceRecognitionTriton()

    def detect_faces(self, image_matrix, detection_threshold: float):
        """
        Detect faces in one image.
        Args:
            image_matrix: (np.ndarray) with shape (H x W x 3)
        Returns:
            bounding_boxes: (np.ndarray) of int in format XYXY with shape (N x 4)
            landmarks: (np.ndarray) of int in format (XY XY XY XY XY) with shape (N x 10)
            confidences: (np.ndarray) of float with shape (N, 1)
        """
        img_height, img_width, _ = image_matrix.shape

        bboxes, kpss = self.face_detector.detect(image_matrix, detection_threshold / 100, input_size=(640, 640))
        return bboxes[:, 0:4].astype(np.integer), kpss.reshape(-1, 10).astype(np.integer), bboxes[:, 4].reshape(-1, 1)

    def extract_features(self, batch_image):
        """
        Extract feature from aligned face.
        Args:
            batch_image: (np.ndarray) with shape (Batch x H x W x 3)
        Returns:
            feature_vectors: (np.ndarray) of float32 with shape (Batch x 512)
        """
        batch_image = (batch_image / 255.0 - 0.5) / 0.5
        img = np.transpose(batch_image, (0, 3, 1, 2))
        feature_vectors = self.face_recognition.extract_features(img)
        feature_vectors = feature_vectors / norm(feature_vectors, axis=1).reshape(-1, 1)
        return feature_vectors
