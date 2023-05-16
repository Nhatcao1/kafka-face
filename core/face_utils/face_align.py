import cv2
import numpy as np
from skimage import transform as trans

src1 = np.array([
    [51.642, 50.115],
    [57.617, 49.990],
    [35.740, 69.007],
    [51.157, 89.050],
    [57.025, 89.702]], dtype=np.float32)

# <--left
src2 = np.array([
    [45.031, 50.118],
    [65.568, 50.872],
    [39.677, 68.111],
    [45.177, 86.190],
    [64.246, 86.758]], dtype=np.float32)

# ---frontal
src3 = np.array([
    [39.730, 51.138],
    [72.270, 51.138],
    [56.000, 68.493],
    [42.463, 87.010],
    [69.537, 87.010]], dtype=np.float32)

# -->right
src4 = np.array([
    [46.845, 50.872],
    [67.382, 50.118],
    [72.737, 68.111],
    [48.167, 86.758],
    [67.236, 86.190]], dtype=np.float32)

# -->right profile
src5 = np.array([
    [54.796, 49.990],
    [60.771, 50.115],
    [76.673, 69.007],
    [55.388, 89.702],
    [61.257, 89.050]], dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2}

arcface_src = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]], dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)


# lmk is prediction, src is template
def estimate_norm(lmk, image_size=112, mode='arcface'):
    assert lmk.shape == (5, 2)
    similarity_transform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_m = []
    min_index = []
    min_error = float('inf')
    if mode == 'arcface':
        assert image_size == 112
        source = arcface_src
    else:
        source = src_map[image_size]
    for i in np.arange(source.shape[0]):
        similarity_transform.estimate(lmk, source[i])
        M = similarity_transform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - source[i]) ** 2, axis=1)))
        # print(error)
        if error < min_error:
            min_error = error
            min_m = M
            min_index = i
    return min_m, min_index


def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M, pose_index = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


def align_faces(img, bboxes, landmarks, max_num=0):
    if bboxes.shape[0] == 0:
        return []
    if 0 < max_num < bboxes.shape[0]:
        area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        img_center = img.shape[0] // 2, img.shape[1] // 2
        offsets = np.vstack(
            [(bboxes[:, 0] + bboxes[:, 2]) / 2 - img_center[1], (bboxes[:, 1] + bboxes[:, 3]) / 2 - img_center[0]])
        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
        box_index = np.argmax(area - offset_dist_squared * 2.0)  # some extra weight on the centering
        box_index = box_index[0:max_num]
        bboxes = bboxes[box_index, :]
        landmarks = landmarks[box_index, :]
    aligned_matrix = None
    for i in range(bboxes.shape[0]):
        landmark = landmarks[i]
        _img = norm_crop(img, landmark=landmark)
        n_img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        aligned_face = np.transpose(n_img, (2, 0, 1))
        if aligned_matrix is None:
            aligned_matrix = np.expand_dims(aligned_face, axis=0)
        else:
            aligned_matrix = np.append(aligned_matrix, np.expand_dims(aligned_face, axis=0), axis=0)

    return aligned_matrix
