import cv2
from producer import ProducerThread
from multiprocessing import Process, freeze_support
from config import *

def start_producer(name, video_path):
    producer = ProducerThread(name, video_path)
    producer.start()

if __name__ == '__main__':
    freeze_support()

    site_1 = ("site_1", "videos/Robert.mov")
    site_2 = ("site_2", "videos/Emma.mov")
    site_3 = ("site_3", 0)

    process1 = Process(target=start_producer, args=site_1)
    process2 = Process(target=start_producer, args=site_2)
    process3 = Process(target=start_producer, args=site_3)

    process1.start()
    process2.start()
    process3.start()

    process1.join()
    process2.join()
    process3.join()