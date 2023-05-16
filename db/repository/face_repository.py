import json
from typing import List, Optional

from domain.face import Face


def save_face_tokens(
        connection,
        faces: List[Face],
        faceset_tokens: List[str],
        created_by: str
) -> bool:
    try:
        cur = connection.cursor()
        for face in faces:
            if face.face_attributes is None:
                face_attributes = []
            else:
                face_attributes = [json.dumps(face_attribute) for face_attribute in face.face_attributes]
            cur.execute(
                "INSERT INTO face("
                "face_token, "
                "person_id, "
                "detection_score, "
                "bounding_box, "
                "landmarks, "
                "feature_vector, "
                "face_attributes, "
                "image_url, "
                "created_by, "
                "updated_by) VALUES  (%s,%s,%s,%s,%s::json[],%s,%s,%s,%s,%s);",
                (face.face_token,
                 face.person_id,
                 face.detection_score,
                 json.dumps(face.bounding_box),
                 [json.dumps(landmark) for landmark in face.landmarks],
                 face.feature_vector,
                 face_attributes,
                 face.image_url,
                 created_by,
                 created_by
                 )
            )

            for faceset_token in faceset_tokens:
                cur.execute("INSERT INTO face_in_faceset (faceset_token, face_token) VALUES (%s, %s);",
                            (faceset_token, face.face_token))
        connection.commit()
        if cur:
            cur.close()
        return True

    except Exception as e:
        print(e)
        return False
    finally:
        if cur:
            cur.close()


def get_person_id_from_face_token(
        connection,
        face_token: str
) -> Optional[str]:
    try:
        cur = connection.cursor()
        cur.execute("SELECT person_id FROM face WHERE face_token=%s;", (face_token, ))
        person_id = cur.fetchone()
        if person_id is not None:
            return person_id[0]
        return person_id
    except Exception as e:
       pass
    finally:
        if cur is not None:
            cur.close()


def get_image_url_from_face_token(
        connection,
        face_token: str
) -> Optional[str]:
    try:
        cur = connection.cursor()
        cur.execute("SELECT image_url FROM face WHERE face_token=%s;", (face_token, ))
        image_url = cur.fetchone()
        if image_url is not None:
            return image_url[0]
        return image_url
    except Exception as e:
       pass
    finally:
        if cur is not None:
            cur.close()
