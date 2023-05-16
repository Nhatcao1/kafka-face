import threading
from confluent_kafka import Consumer
from pymongo import MongoClient
from face_recognition import recognition
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

        # rgb = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        person_name = recognition(image)

        self.update_database(person_name[0])
            

    def update_database(self, name):
        print("Updating absense employee: ", name)
        collection = self.db["Employee"]
        myquery = { "name": name}
        newvalues = { "$set": { "absense": True}}
        collection.update_one(myquery, newvalues)
        print("update database sucessfull")

    
    def start(self, numThreads):
        while True: 
            self.read_data()


