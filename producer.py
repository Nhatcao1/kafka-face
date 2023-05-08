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

    def publishFrame(self):
        print("RUNNING PUBLISH FRAME")
        # take single image add feed to consumer for personal detect
        video = cv2.VideoCapture(self.video_path)
        # video_name = os.path.basename(self.video_path).split(".")[0]
        frame_no = 1

        #Merge several Image video
        while video.isOpened():
            _, frame = video.read()
            # print("reading frame")
            # pushing every 5rd frame
            if frame is None: 
                break
            if frame_no % 10 == 0:
                # shoot a single image
                frame = cv2.resize(frame, (150, 150), interpolation = cv2.INTER_AREA)
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
