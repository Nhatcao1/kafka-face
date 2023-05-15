import numpy as np
import cv2
from core.face_utils import face_align


def align_face(image_matrix, landmark, output_size=112, mode="arcface"):
    """
    Align face to 112 x 112
    Args:
        image_matrix: np.ndarray image with shape H x W x 3 (Entire image)
        landmark: np.ndarray of int in XY format with shape (10, )
        output_size: int expected output size, default to 112 (arcface)
        mode: align mode, "arcface" vs "others"
    Returns:
        cropped_image: np.ndarray of cropped image in RGB
    """
    point = landmark.reshape((-1, 2))
    cropped_image = face_align.norm_crop(image_matrix, landmark=point, image_size=output_size, mode=mode)
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    return cropped_image


def pad_input_image(img, max_steps):
    """
        pad image to suitable shape
    """
    img_h, img_w, _ = img.shape
    img_pad_h = 0
    if img_h % max_steps > 0:
        img_pad_h = max_steps - img_h % max_steps

    img_pad_w = 0
    if img_w % max_steps > 0:
        img_pad_w = max_steps - img_w % max_steps
    pad_val = np.mean(img, axis=(0, 1)).astype(np.uint8)
    img = cv2.copyMakeBorder(img, 0, img_pad_h, 0, img_pad_w, cv2.BORDER_CONSTANT, value=pad_val.tolist())
    pad_params = (img_h, img_w, img_pad_h, img_pad_w)
    return img, pad_params


def recover_pad_output(outputs, pad_params):
    """recover the padded output effect"""
    img_h, img_w, img_pad_h, img_pad_w = pad_params
    recover_xy = np.reshape(outputs[:, :14], [-1, 7, 2]) * [(img_pad_w + img_w) / img_w, (img_pad_h + img_h) / img_h]
    outputs[:, :14] = np.reshape(recover_xy, [-1, 14])

    return outputs