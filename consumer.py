import threading
from confluent_kafka import Consumer
from pymongo import MongoClient
import face_recognition
import pickle
import cv2
import numpy as np
import time

class ConsumerThread:
    def __init__(self, config, topic_list):
        self.config = config
        self.db = MongoClient('mongodb://localhost:27017')["test"]
        # this consumer meant to take list of kafka
        self.topic = topic_list
        # getting recognizer
        self.detection_method = "hog" # or cnn
        self.employee_list = {}
        self.consumer = Consumer(self.config)
        self.encodings = "weight/encodings.pickle"
       
    
    def get_name_from_database(self):
        collection = self.db["EmployeeID"]
        column_data = collection.find({}, {"id": 1, "name": 1})
        for column in column_data:
            self.employee_list[column["id"]] = column["name"]
        print(self.employee_list)
        

    def read_data(self):
        # consumer subcribe topic list
        self.consumer.subscribe(self.topic)
        print(self.topic)
        print("consuming data start")
        img = None
        while True:
            #fetch data assign to topic
            event = None
            try:
                event = self.consumer.poll(0.5)
                print(self.topic)
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
                print(type(img))
                #run on different thread, machine 
                self.face_recognition(img)
        

    def face_recognition(self,image):
        print("Recognizing a face")
        data = pickle.loads(open(self.encodings, "rb").read())
        # rgb = image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # detect the (x, y)-coordinates of the bounding boxes corresponding
        # to each face in the input image, then compute the facial embeddings
        # for each face
        print("[INFO] recognizing faces...")
        boxes = face_recognition.face_locations(rgb,model=self.detection_method)
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

            # update the list of names
            names.append(name)
            
            self.update_database(names[0])
            

    def update_database(self, name):
        print("Updating absense employee: ", name)
        collection = self.db["Employee"]
        myquery = { "name": name}
        newvalues = { "$set": { "absense": True}}
        collection.update_one(myquery, newvalues)
        print("update database sucessfull")
        
        # kill comsumer after update database
        self.consumer.close()
    
    def start(self, numThreads):
        # Note that number of consumers in a group shouldn't exceed the number of partitions in the topic
        # for _ in range(numThreads):
        #     t = threading.Thread(target=self.read_data)
        #     t.daemon = True
        #     t.start()
        while True: 
            self.read_data()
        # 

    # topic = ["multi-video-stream"]
    # consumer_thread = ConsumerThread(consumer_config, topic, 32, model, db, videos_map)
    # consumer_thread.start(3)

