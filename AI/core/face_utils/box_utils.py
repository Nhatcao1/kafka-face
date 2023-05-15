import numpy as np
from log import logger


def sort_by_size(bounding_boxes, landmarks=None, confidences=None, limit=None):
    """
    Sort boxes (landmarks, confidences) by size of the bounding_boxes in desc order
    Args:
        bounding_boxes: np.ndarray in (x,y,w,h) format with shape (N x 4)
        landmarks: np.ndarray with shape (N x 10)
        confidences: np.ndarray with shape (N x 1)
        limit: (int) maximum of number to be returned, default to None = return all boxes
    Returns:
        sorted_bounding_boxes: np.ndarray in (x,y,w,h) format with shape (N x 4)
        sorted_landmarks: default = None
        sorted_confidences: default = None
    """
    sorted_index = np.argsort(
        (bounding_boxes[:, 2] - bounding_boxes[:, 0]) * (bounding_boxes[:, 3] - bounding_boxes[:, 1]))[::-1]

    # Limit number of faces
    bounding_boxes = bounding_boxes[sorted_index]
    if landmarks is not None:
        landmarks = landmarks[sorted_index]
    if confidences is not None:
        confidences = confidences[sorted_index]

    return bounding_boxes[:limit], landmarks[:limit], confidences[:limit]


def get_coordinates_with_margin(image, bbox, margin: int = -1):
    """
    Add margin to box
    Args:
        image: np.ndarray of image with shape H x W x C
        bbox: XYXY coordinates
        margin: margin to be add
    Returns:
        [new_x1, new_y1, new_x2, new_y2]
    """
    if margin < 0:
        margin = (bbox[2] - bbox[0]) // 4

    x1 = max(0, bbox[0] - margin)
    x2 = min(image.shape[1], bbox[2] + margin)
    y1 = max(0, bbox[1] - margin)
    y2 = min(image.shape[0], bbox[3] + margin)
    return x1, y1, x2, y2


def expand_coordinates(frame, bbox, expand_ratio=1.0):
    """
    Add margin to box
    Args:
        frame: np.ndarray of image with shape H x W x C
        bbox: XYXY coordinates
        expand_ratio: ratio to expands
    Returns:
        [new_x1, new_y1, new_x2, new_y2]
    """
    if expand_ratio == 1.:
        return bbox

    x1, y1, x2, y2 = bbox
    new_width = (x2 - x1) / 2 * expand_ratio
    new_height = (y2 - y1) / 2 * expand_ratio
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    new_y1 = max(0, int(y_center - new_height))
    new_y2 = min(frame.shape[0], int(y_center + new_height))
    new_x1 = max(0, int(x_center - new_width))
    new_x2 = min(frame.shape[1], int(x_center + new_width))
    return [new_x1, new_y1, new_x2, new_y2]


def expand_coordinates_1(frame, bbox, expand_ratio=1.0):
    """
    Add margin to box
    Args:
        frame: np.ndarray of image with shape H x W x C
        bbox: XYXY coordinates
        expand_ratio: ratio to expands
    Returns:
        [new_x1, new_y1, new_x2, new_y2]
    """

    if expand_ratio == 1.:
        return bbox

    x1, y1, x2, y2 = bbox
    new_width = (x2 - x1) * expand_ratio
    new_height = (y2 - y1) * expand_ratio
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    if new_width / new_height >= 0.75:
        new_height = new_width / 0.75
    else:
        new_width = new_height * 0.75

    if y_center <= new_height:
        new_y1 = 0
        new_y2 = min(frame.shape[0], int(2 * new_height))

    elif y_center + new_height >= frame.shape[0]:
        new_y1 = max(0, int(frame.shape[0] - 2 * new_height))
        new_y2 = int(frame.shape[0])
    else:
        new_y1 = max(0, int(y_center - new_height))
        new_y2 = min(frame.shape[0], int(y_center + new_height))

    if x_center <= new_width:
        new_x1 = 0
        new_x2 = min(frame.shape[1], int(2 * new_width))
    elif x_center + new_width >= frame.shape[1]:
        new_x1 = max(0, int(frame.shape[1] - 2 * new_width))
        new_x2 = int(frame.shape[1])
    else:
        new_x1 = max(0, int(x_center - new_width))
        new_x2 = min(frame.shape[1], int(x_center + new_width))
    return [new_x1, new_y1, new_x2, new_y2]


def clip_box(im_height, im_width, box):
    """
    Inplace clip box to [0, im_height / im_width]
    Args:
        im_height (int): maximum height
        im_width (int): maximum width
        box: list or numpy array with shape (4,) [x1, y1, x2, y2]

    Returns:
        clipped_box: np
    """
    box[0::2] = np.clip(box[0::2], 0, im_width)
    box[1::2] = np.clip(box[1::2], 0, im_height)
    return box
