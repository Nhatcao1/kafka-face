import threading
from confluent_kafka import Consumer, KafkaError, KafkaException
from pymongo import MongoClient

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
        self.machine_learning = cv2.CascadeClassifier('weight/haarcascade_frontalface_alt_tree.xml')
        self.employee_list = {}
        self.consumer = Consumer(self.config)

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read('weight/trainer.yml')
       
    
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
                print(type(img))
                #run on different thread, machine 
                # self.face_recognition(img)
                break

        if img is None:
            print("No face")
        else:
            print("Detected a face, recognizing process")
            self.face_recognition(img)
        # time.sleep(20)
    #connect database

    def face_recognition(self,image):
        print("Recognizing a face")
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        faces = self.machine_learning.detectMultiScale( gray,scaleFactor = 1.2,minNeighbors = 5)
        print("How many face:", len(faces))
        for(x,y,w,h) in faces:
            id, confidence = self.recognizer.predict(gray[y:y+h,x:x+w])
            print("id is :", id)
            print("confidence score:",confidence)
            # If confidence is less them 100 ==> "0" : perfect match 
            if (confidence < 100):
                print("Face recognized updating database")
                id = self.employee_list[id] # data base and stuff
                self.update_database(id)
                # confidence = "  {0}%".format(round(100 - confidence))
            else:
                print("Stranger Danger")

    def update_database(self, name):
        print("Updating absense employee: ", name)
        myquery = { "name": name}
        newvalues = { "$set": { "absense": True} }
        self.db.update_one(myquery, newvalues)
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

