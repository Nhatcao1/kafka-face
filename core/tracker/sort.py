"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import time

import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

from filterpy.kalman import KalmanFilter

from typing import List
from domain.face import FaceInfo

np.random.seed(0)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, face_info: FaceInfo, init_time: int):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(face_info.bounding_box)
        self.time_since_update = 0
        # Track id, set maximum to 10^9. Track id reset after 10^9 records
        self.id: int = KalmanBoxTracker.count % 1000000000
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        # Added attributes
        self.init_time = init_time
        self.detection_index: int = -1
        self.face_info = face_info
        # Mask
        self.mask_score: float = -1.0

    def update(self, face_info: FaceInfo, recognition_threshold):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(face_info.bounding_box))
        self.update_face_info(face_info=face_info, recognition_threshold=recognition_threshold)

    def update_face_info(self, face_info, recognition_threshold):
        if self.face_info.recognize_confidence < recognition_threshold:
            self.face_info.feature_vector = face_info.feature_vector
        self.face_info.bounding_box = face_info.bounding_box
        self.face_info.landmarks = face_info.landmarks
        self.face_info.detection_score = face_info.detection_score

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


class Sort(object):
    def __init__(self, max_age=5, min_hits=3, max_distance=0.9, recognition_threshold=0.72):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_distance = max_distance
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        self.recognition_threshold = recognition_threshold

    def update(self, faces_info: List[FaceInfo], timestamp=int(time.time() * 1000)) -> List[KalmanBoxTracker]:
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame
        even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(faces_info, self.trackers, self.max_distance)
        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(faces_info[m[0]], self.recognition_threshold)
            # set detection_index to map with other detection values such as bounding boxes, landmarks, key points ...
            self.trackers[m[1]].detection_index = m[0]
        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(faces_info[i], timestamp)
            trk.detection_index = i
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
                ret.append(trk)
            i -= 1

            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                del self.trackers[i]
        # Returned ret contain either empty list or list of KalmanBoxTracker having time_since_update < 1
        return ret


def _get_iou(bbox, candidates):

    bbox_tl, bbox_br = bbox[:2], bbox[2:]
    candidates_tl, candidates_br = candidates[:, :2], candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def get_iou_cost_matrix(bb_detects, bb_tracks,) -> np.ndarray:
    cost_matrix = np.zeros((len(bb_tracks), len(bb_detects)))
    for row, bb_track in enumerate(bb_tracks):
        cost_matrix[row, :] = 1. - _get_iou(bb_track, bb_detects)
    return cost_matrix


def get_euclid_distance_cost_matrix(faces_info: List[FaceInfo], trackers: List[KalmanBoxTracker], max_distance=100):
    cp_detects = []
    cp_tracks = []

    for face_info in faces_info:
        bbox = face_info.bounding_box
        cp_detects.append([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])

    for tracker in trackers:

        bbox = tracker.face_info.bounding_box
        cp_tracks.append([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])

    cp_detects = np.array(cp_detects)
    cp_tracks = np.array(cp_tracks)
    cost_matrix = np.zeros((len(cp_detects), len(cp_tracks)))

    for row, _ in enumerate(cost_matrix):
        cost_matrix[row, :] = np.sqrt(np.power(cp_detects[row, 0] - cp_tracks[:, 0], 2) +
                                      np.power(cp_detects[row, 1] - cp_tracks[:, 1], 2))
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    return cost_matrix


def _cosine_distance(row_array, col_array, data_is_normalized=False):
    if not data_is_normalized:
        row_array = np.asarray(row_array) / np.linalg.norm(row_array, axis=1, keepdims=True)
        col_array = np.asarray(col_array) / np.linalg.norm(col_array, axis=1, keepdims=True)
    return 1. - np.dot(row_array, col_array.T)


def _nn_cosine_distance(row_array, col_array):
    distances = _cosine_distance(row_array, col_array)
    return distances.min(axis=0)


def get_feature_cost_matrix(faces_info: List[FaceInfo], trackers: List[KalmanBoxTracker]) -> np.ndarray:
    cost_matrix = np.zeros((len(faces_info), len(trackers)))
    features_detections = np.array([[face_info.feature_vector] for face_info in faces_info])
    features_track = np.array([tracker.face_info.feature_vector for tracker in trackers])

    for idx, features_detection in enumerate(features_detections):
        cost_matrix[idx, :] = _nn_cosine_distance(row_array=features_detection, col_array=features_track)
    return cost_matrix


def associate_detections_to_trackers(faces_info: List[FaceInfo], trackers: List[KalmanBoxTracker], max_distance=0.9):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(faces_info)), np.empty((0, 5), dtype=int)
    if len(faces_info) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(faces_info)), np.arange(len(trackers), dtype=int)

    cost_matrix = (0.5/40) * get_euclid_distance_cost_matrix(faces_info=faces_info, trackers=trackers, max_distance=40)
    cost_matrix += 0.5 * get_feature_cost_matrix(faces_info=faces_info, trackers=trackers)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    try:
        row_indices, col_indices = linear_assignment(cost_matrix)
    except Exception as e:
        print(e)
        return np.empty((0, 2), dtype=int), np.arange(len(faces_info)), np.arange(len(trackers), dtype=int)
    unmatched_detections, unmatched_tracks, matches = [], [], []

    for col, track in enumerate(trackers):
        if col not in col_indices:
            unmatched_tracks.append(col)

    for row, face_info in enumerate(faces_info):
        if row not in row_indices:
            unmatched_detections.append(row)

    for row, col in zip(row_indices, col_indices):
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(col)
            unmatched_detections.append(row)
        else:
            matches.append((row, col))
    return np.array(matches), np.array(unmatched_detections), np.array(unmatched_tracks)
