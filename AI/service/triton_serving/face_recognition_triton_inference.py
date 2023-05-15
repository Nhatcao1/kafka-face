import tritonclient.grpc as grpcclient
import numpy as np

from core.config import get_app_settings


class FaceRecognitionTriton:
    def __init__(self):
        settings = get_app_settings()
        self.triton_client = grpcclient.InferenceServerClient(
            url=settings.triton_host,
            verbose=False,
            ssl=False,
            root_certificates=None,
            private_key=None,
            certificate_chain=None
        )

        self.input_name = "input.1"
        self.output_name = '1333'
        self.model_name = "pytorch_resnet100_onnx"

    def extract_features(self, images_matrix):
        inputs = []
        batch_size = images_matrix.shape[0]
        inputs.append(grpcclient.InferInput(self.input_name,
                                            [batch_size, 3, 112, 112], "FP32"))

        inputs[0].set_data_from_numpy(images_matrix.astype(np.float32))

        outputs = [grpcclient.InferRequestedOutput(self.output_name)]

        result = self.triton_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs,
            client_timeout=None,
            headers={'test': '1'},  # Not sure if this is necessary
            compression_algorithm=None
        )

        return result.as_numpy(self.output_name)
