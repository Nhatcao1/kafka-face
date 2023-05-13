from confluent_kafka import Consumer, Producer
import face_recognition
import pickle
import cv2
import numpy as np
from config import *
import datetime

class ConsumerThread:
    def __init__(self, config, topic_list, site_number):
        self.config = config
        # this consumer meant to take list of kafka
        self.topic = topic_list
        # getting recognizer
        self.employee_list = {}
        self.consumer = Consumer(self.config)
        self.site_number = site_number
        self.absence_status = absence_status
        self.authorization = authorization
        self.true_false = False # don't recognise face then recognise

    ######new_code########
    def SingleShotProducer(self, name, topic):
        message = "Warning! UnKnown person at " + str(self.site_number)
        # print(message)
        if name != "Unknown" and self.authorization[name][self.site_number - 1]:
            message = "Employee " + name + " authorized to site: " + str(self.site_number)
        elif name != "Unknown" and self.authorization[name][self.site_number - 1] is False:
            message = "Employee " + name + ". You are not authorized to site: " + str(self.site_number)
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
    def log_entrance_event(self, employee_name, time_at_entrance, image):
        if self.authorization[employee_name][self.site_number - 1] is False:
            return
        #upgrade MongoDB
        mongodb_id = employee_name + "_" + str(time_at_entrance) + "_site" + str(self.site_number)
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
        cursor.execute(insert_query, (employee_name, time_at_entrance, self.site_number, mongodb_id))
        postgres_conn.commit()
        cursor.close()
        print("Entrance event logged successfully!")

    def log_absence(self, employee_name, absense_status=False):
        if self.authorization[employee_name][self.site_number - 1] is False:
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
                # print(type(img))
                #run on different thread, machine 
                self.face_recognition(img)
        

    def face_recognition(self,image):
        # print("Recognizing a face")
        data = pickle.loads(open(pickle_encodings, "rb").read())
        # rgb = image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # detect the (x, y)-coordinates of the bounding boxes corresponding
        # to each face in the input image, then compute the facial embeddings
        # for each face
        print("[INFO] recognizing faces...")
        boxes = face_recognition.face_locations(rgb,model=detection_method)
        encodings = face_recognition.face_encodings(rgb, boxes)
        # initialize the list of names for each face detected
        names = []
        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                # determine the recognized face with the largest number of
                # votes (note: in the event of an unlikely tie Python will
                # select first entry in the dictionary)
                name = max(counts, key=counts.get)
                #supposed that every turn, only one face
                names.append(name)

        # update the list of names
        # names.append(name)
        if len(names) != 0:
            for name in names:
                if self.absence_status[name] == False:
                    # self.absence_status[name] = True
                    self.true_false = False
                    timestamp = datetime.datetime.now()
                    self.log_entrance_event(employee_name=name, time_at_entrance=timestamp, image = rgb)
                    self.log_absence(employee_name=name)
                    self.SingleShotProducer(name, self.topic[0])
        elif self.true_false == False:
            self.true_false = True
            self.SingleShotProducer("Unknown", self.topic[0])

# https://stackoverflow.com/questions/49493493/python-store-cv-image-in-mongodb-gridfs


# UPDATE absense
# SET absense_status = True
# WHERE employee_name = "Nhatcao";