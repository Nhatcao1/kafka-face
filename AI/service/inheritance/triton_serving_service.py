from core.config import get_app_settings
from log import logger

import numpy as np
from numpy.linalg import norm
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from core.face_utils.scrfd_processing import ScrfdProcessing


settings = get_app_settings()


class TritonServingService:
    def __init__(self):
        super().__init__()
        
        self.triton_client = grpcclient.InferenceServerClient(
            url=settings.triton_host,
            verbose=False,
            ssl=False,
            root_certificates=None,
            private_key=None,
            certificate_chain=None
        )
        self.face_detection_processing = ScrfdProcessing()



    def get_triton_client(self, triton_host):
        self.triton_client = grpcclient.InferenceServerClient(
                url=triton_host,
                verbose=False,
                ssl=False,
                root_certificates=None,
                private_key=None,
                certificate_chain=None
            )

    def get_triton_infer_output(self, model_name, inputs, outputs):
        try:
            result = self.triton_client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs,
                client_timeout=None,
                headers={'test': '1'},  # Not sure if this is necessary
                compression_algorithm=None)
        except InferenceServerException as e:
            print("Triton Connection Refused")
            print(f"exception: {e}")
            self.get_triton_client(triton_host=settings.triton_host)
            result = None
        return result

    def detect_faces(self, image_matrix, detection_threshold: float):
        """
        Detect faces in one image.
        Args:
            image_matrix: (np.ndarray) with shape (H x W x 3)
            detection_threshold: detection threshold [0-100]
        Returns:
            bounding_boxes: (np.ndarray) of int in format XYXY with shape (N x 4)
            landmarks: (np.ndarray) of int in format (XY XY XY XY XY) with shape (N x 10)
            confidences: (np.ndarray) of float with shape (N, 1)
        """
        model_name = "scrfd"
        inputs, outputs = self.face_detection_processing.pre_processing(image_matrix, input_size=(640, 640))
        results = self.get_triton_infer_output(model_name=model_name, inputs=inputs, outputs=outputs)
        bboxes, kpss = np.empty((1, 5)), np.empty((1, 10))
        if results is not None:
            bboxes, kpss = self.face_detection_processing.post_processing(results, image_matrix, detection_threshold/100)
        return bboxes[:, 0:4].astype(np.integer), kpss.reshape(-1, 10).astype(np.integer), bboxes[:, 4].reshape(-1, 1)

    def extract_features(self, batch_image):
        """
        Extract feature from aligned face.
        Args:
            batch_image: (np.ndarray) with shape (Batch x H x W x 3)
        Returns:
            feature_vectors: (np.ndarray) of float32 with shape (Batch x 512)
        """

        input_name = "0"
        output_name = '926'
        model_name = "mask_recognition"
        batch_image = np.transpose(batch_image, (0, 3, 1, 2))
        inputs = []
        batch_size = batch_image.shape[0]
        inputs.append(grpcclient.InferInput(input_name,
                                            [batch_size, 3, 112, 112], "FP32"))
        inputs[0].set_data_from_numpy(batch_image.astype(np.float32))
        outputs = [grpcclient.InferRequestedOutput(output_name)]
        results = self.get_triton_infer_output(model_name, inputs, outputs)
        feature_vectors = np.empty((batch_size, 512))
        if results is not None:
            feature_vectors = results.as_numpy(output_name)
            feature_vectors = feature_vectors / norm(feature_vectors, axis=1).reshape(-1, 1)
        return feature_vectors






