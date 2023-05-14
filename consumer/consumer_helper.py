from confluent_kafka import Consumer, Producer
import face_recognition
import pickle
import cv2
import numpy as np
from config import *
import datetime

def SingleShotProducer(site_number, name, topic):
        message = "Warning! UnKnown person at " + str(site_number)
        # print(message)
        if name != "Unknown" and authorization[name][site_number - 1]:
            message = "Employee " + name + " authorized to site: " + str(site_number)
        elif name != "Unknown" and authorization[name][site_number - 1] is False:
            message = "Employee " + name + ". You are not authorized to site: " + str(site_number)
        # print(message)
        # the topic for return channel have _return
        if name != "Unknown" and absence_status[name] == True:
            return
        absence_status[name] = True
        producer = Producer(producer_config)
        producer.produce(topic + "_return", value = message)
        producer.flush()
        # producer.close()

# Task to update PostgreSQL
# Function to log entrance event
def log_entrance_event(site_number, employee_name, time_at_entrance, image):
    if authorization[employee_name][site_number - 1] is False:
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

    def log_absence(self, employee_name, absense_status=False):
        if authorization[employee_name][site_number - 1] is False:
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

    def read_data():
        # consumer subcribe topic list
        consumer.subscribe(topic)
        print("consuming data start")
        print(topic)
        img = None
        while True:
            #fetch data assign to topic
            event = None
            try:
                event = consumer.poll(0.5)
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
                face_recognition(img)
        

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
                if absence_status[name] == False:
                    # absence_status[name] = True
                    true_false = False
                    timestamp = datetime.datetime.now()
                    log_entrance_event(employee_name=name, time_at_entrance=timestamp, image = rgb)
                    log_absence(employee_name=name)
                    SingleShotProducer(name, topic[0])
        elif true_false == False:
            true_false = True
            SingleShotProducer("Unknown", topic[0])