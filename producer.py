from confluent_kafka import Producer
from config import producer_config
import cv2
import numpy as np
from utils import *
import os
import time
# import concurrent.futures

class ProducerThread:
    def __init__(self, config, topic_name, video_path):
        self.producer = Producer(config)
        self.topic_name = topic_name
        self.video_path = video_path
        self.machine_learning = cv2.CascadeClassifier('weight/haarcascade_frontalface_alt_tree.xml')

    def publishFrame(self):
        print("RUNNING PUBLISH FRAME")
        #https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
        # take single image add feed to consumer for personal detect
        video = cv2.VideoCapture(self.video_path)
        # video_name = os.path.basename(self.video_path).split(".")[0]
        frame_no = 1

        #Merge several Image video
        while video.isOpened():
            _, frame = video.read()
            # print("reading frame")
            # pushing every 3rd frame, check if there is a face
            if frame_no % 3 == 0:
                # we will send this as the fine grain original in jpeg 
                # light weigth face detect
                # img = cv2.flip(frame, -1)
                # print(img)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.machine_learning.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5)
                # print("PRODUCER 'S FACES: ", len(faces))
                if len(faces) > 0:
                    print("detect a face")
                    # shoot a single image
                    try:
                        self.producer.produce(
                            topic=self.topic_name,
                            value=serializeImg(frame),
                            # value=gray, 
                            on_delivery= delivery_report, #  heavy
                        )
                        self.producer.poll(0) # send with zero time out
                        print("send to broker")
                    except Exception as e:
                        print(e)
            time.sleep(0.1)
            frame_no = frame_no + 1
        video.release()
        return
        
    def start(self):
        # runs until the processes in all the threads are finished
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     executor.submit(self.publishFrame)
        #     # executor.map(self.publishFrame, self.video_path)

        self.publishFrame()
        self.producer.flush() # push all the remaining messages in the queue
        print("Finished...")
