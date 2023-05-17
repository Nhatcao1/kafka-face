import threading
from confluent_kafka import Consumer, Producer
import face_recognition
import pickle
import cv2, re
import numpy as np
from config import *
import datetime
from core.config import get_app_settings
from core.face_utils.visualize import draw_detected_faces

from core.tracker.sort import Sort

from domain.face import FaceInfo
from service.face_image_service import FaceImageService
import time

class ConsumerThread:
    def __init__(self, config, topic_list):
        self.config = config
        # this consumer meant to take list of kafka
        self.topic = topic_list
        # getting recognizer
        self.employee_list = {}
        self.consumer = Consumer(self.config)
        self.absence_status = absence_status
        self.authorization = authorization
        self.true_false = False # don't recognise face then recognise
        self.settings = get_app_settings()
        
    ######new_code########
    def SingleShotProducer(self, name, topic, site_number):
        message = "Warning! UnKnown person at " + topic
        # print(message)
        if name != "Unknown" and self.authorization[name][site_number - 1]:
            message = "Employee " + name + " authorized to site: " + str(site_number)
        elif name != "Unknown" and self.authorization[name][site_number - 1] is False:
            message = "Employee " + name + ". You are not authorized to site: " + str(site_number)
        # print(message)
        # the topic for return channel have _return
        if name != "Unknown" and self.absence_status[name] == True:
            return
        self.absence_status[name] = True
        producer = Producer(producer_config)
        producer.produce(topic + "_return", value = message)
        producer.flush()
        # producer.close()

    # Task to update PostgreSQL
    # Function to log entrance event
    def log_entrance_event(self, employee_name, time_at_entrance, image, site_number):
        if self.authorization[employee_name][site_number - 1] is False:
            return
        #upgrade MongoDB
        mongodb_id = employee_name + "_" + str(time_at_entrance) + "_site" + str(site_number)
        mongo_image = fs.put(image.tostring(), encoding='utf-8') # store image to fs

        document = {
            'image_id': mongodb_id,
            'image': mongo_image, #image metadata
            'shape': image.shape
        }

        mongo_db["image_store"].insert_one(document)

        ####update postgres
        cursor = postgres_conn.cursor()
        # timestamp = datetime.now()
        insert_query = """
        INSERT INTO entrance_log (employee_name, time_at_entrance, site, unique_id_link)
        VALUES (%s, %s, %s, %s);
        """
        cursor.execute(insert_query, (employee_name, time_at_entrance, site_number, mongodb_id))
        postgres_conn.commit()
        cursor.close()
        print("Entrance event logged successfully!")

    def log_absence(self, employee_name, site_number, absense_status=False):
        if self.authorization[employee_name][site_number - 1] is False:
            return
        cursor = postgres_conn.cursor()
        update_query = """
        UPDATE absense
        SET absense_status = %s
        WHERE employee_name = %s;
        """
        cursor.execute(update_query, (absense_status ,employee_name))
        postgres_conn.commit()
        cursor.close()
        print("Absense status updated successfully!")

    def read_data(self):
        # consumer subcribe topic list
        self.consumer.subscribe(self.topic)
        print("consuming data start")
        print(self.topic)
        img = None
        while True:
            #fetch data assign to topic
            event = None
            try:
                event = self.consumer.poll(0.5)
            except Exception as e:
                print(e)
            #regard less of topic, consumer will consumer incoming image
            if event is None:
                # print("There is no event")
                continue
            elif event.error():
                print("Consumer error: {}".format(event.error()))
                continue
            elif event.error()==None:
                #DO AI STUFF HERE, and get label stuff, I am so tired
                print("CONSUMING IMAGE")
                nparr = np.frombuffer(event.value(), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                # img = event.value()
                #run on different thread, machine 
                headers = event.headers()
                original = headers[0][1].decode('utf-8')
                self.face_recognition(img,original)
        
    def extract_number(self, string):
        # Use regular expression to find the number at the end of the string
        match = re.search(r'\d+$', string)

        if match:
            return int(match.group())

        # Return None if no number is found
        return None

    def face_recognition(self,image):
        db_pool = psycopg2.pool.ThreadedConnectionPool(
        self.settings.min_connection_count,
        self.settings.max_connection_count,
        # "postgresql://postgres:changeme@localhost:5432/face_management"
        "postgresql://postgres:123@localhost:5432/postgres"
        )
        db_connection = db_pool.getconn()
        face_image_service = FaceImageService()
        recognition_threshold = 75
        detection_threshold = 60
        faceset_token = "_c3770594291f452782520613dfeaa6c"
        tracker = Sort(max_age=3, min_hits=0, max_distance=0.82,
                    recognition_threshold=0.75)

        sent_event_recognized_track_ids = []
        sent_event_unknown_track_ids = []
        timestamp = int(time.time())
        detected_faces, log_message = face_image_service.detect_faces(
            image=frame,
            detection_threshold=detection_threshold,
            upload_cropped_image=False,
            base_image_path="test"
        )
        frame_height, frame_width, _ = frame.shape

        print(len(detected_faces))

        face_image_service.recognition_detected_faces(
            frame=frame,
            timestamp=timestamp,
            detected_faces=detected_faces,
            recognition_threshold=recognition_threshold,
            faceset_token=faceset_token,
            tracker=tracker,
            db_connection=db_connection
        )

        frame = draw_detected_faces(frame, detected_faces, recognition_threshold,
                                    timestamp)
        
        frame,text = cv2.resize(frame, (1280, 720), cv2.INTER_AREA)
        name = text

        if self.absence_status[name] == False:
            # self.absence_status[name] = True
            self.true_false = False
            timestamp = datetime.datetime.now()
            self.log_entrance_event(employee_name=name, time_at_entrance=timestamp, image = rgb, site_number=site_number)
            self.log_absence(employee_name=name, site_number=site_number)
            self.SingleShotProducer(name, original_site, site_number=site_number)
        elif self.true_false == False:
            self.true_false = True
            self.SingleShotProducer("Unknown", original_site, site_number=site_number)