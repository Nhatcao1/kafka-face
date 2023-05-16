from typing import List
from domain.face import FaceInfo
from datetime import datetime


def save_debug(video_path: str, face_infos: List[FaceInfo], timestamp):
    debug_path = video_path.replace("mp4", "txt")
    f = open(debug_path, "a")
    for face in face_infos:
        topK_face_token = ""
        for face_token in face.topK_similar:
            topK_face_token += str(face_token) + "-"
        f.write(f"{datetime.fromtimestamp(timestamp/1000)}|{face.track_id}|{topK_face_token}|{face.person_id}|{face.recognize_confidence}\n")
    f.close()
