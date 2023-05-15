from typing import List

from core.tracker.sort import KalmanBoxTracker
from domain.face import FaceInfo


def update_tracks_info(
        unrecognized_faces: List[FaceInfo],
        tracker
) -> None:
    track_ids_from_unrecognized_faces = [face.track_id for face in unrecognized_faces]
    for track in tracker.trackers:
        if track.id in track_ids_from_unrecognized_faces:
            idx = track_ids_from_unrecognized_faces.index(track.id)
            track.face_info.matched_face_token = unrecognized_faces[idx].matched_face_token
            track.face_info.recognize_confidence = unrecognized_faces[idx].recognize_confidence
            track.face_info.person_id = unrecognized_faces[idx].person_id


def get_info_from_track(detected_faces: List[FaceInfo], active_tracks: List[KalmanBoxTracker]):
    for track in active_tracks:
        idx = track.detection_index
        detected_faces[idx].track_id = track.id
        detected_faces[idx].matched_face_token = track.face_info.matched_face_token
        detected_faces[idx].recognize_confidence = track.face_info.recognize_confidence
        detected_faces[idx].person_id = track.face_info.person_id
        detected_faces[idx].track_init_time = track.init_time
        detected_faces[idx].area_ids = track.face_info.area_ids
        detected_faces[idx].alert = track.face_info.alert
        update_face_attributes = []
        for face_attribute_1, face_attribute_2 in zip(detected_faces[idx].face_attributes,
                                                      track.face_info.face_attributes):
            if face_attribute_1.attribute_confidence <= face_attribute_2.attribute_confidence:
                update_face_attributes.append(face_attribute_2)
            else:
                update_face_attributes.append(face_attribute_1)
        detected_faces[idx].face_attributes = update_face_attributes
        detected_faces[idx].history_face_token = track.face_info.history_face_token
