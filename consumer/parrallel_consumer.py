import sys
sys.path.append('/Users/nhatcao/multi-topics-video-stream')
from celery import Celery, group
from confluent_kafka import Consumer, Producer
from datetime import datetime
import face_recognition
from config import *
from celery_helper import *
import pickle
import cv2
import numpy as np


pickle_encodings = "weight/encodings.pickle"
authorization = {}
absence_status = {}

# Create a Celery app
app = Celery('celery-consumer', broker='amqp://guest@localhost:5672//', backend='rpc://')
# app = Celery('celery-consumer', broker='amqp://guest:guest@rabbitmq:5672//')

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

####
# Define tasks for each topic
@app.task
def consume_topic1():
    print("Consumer online")
    consumer = Consumer(consumer_config)
    consumer.subscribe(["site_1"])
    while True:
        try:
            messages = consumer.poll(0.5)
            if messages is not None:
                for message in messages:
                    process_message(message, "site_1", 1)
        except Exception as e:
            print(e)

@app.task
def consume_topic2():
    print("Consumer online")
    consumer = Consumer(consumer_config)
    consumer.subscribe(["site_2"])
    while True:
        try:
            messages = consumer.poll(0.5)
            if messages is not None:
                for message in messages:
                    process_message(message, "site_2", 2)
        except Exception as e:
            print(e)
        
@app.task
def consume_topic3():
    print("Consumer online")
    consumer = Consumer(consumer_config)
    consumer.subscribe(["site_3"])
    while True:
        try:
            messages = consumer.poll(0.5)
            if messages is not None:
                for message in messages:
                    process_message(message, "site_3", 3)
        except Exception as e:
            print(e)
            
@app.task
def run_parallel_tasks():
    # Create a group of tasks
    task_group = group([consume_topic1.s(), consume_topic2.s(), consume_topic3.s()])

    # Run the tasks in parallel
    result = task_group.apply_async()

    # Wait for the tasks to complete
    task_results = result.get()

    for task_result in task_results:
        if task_result.successful():
            # Task completed successfully
            print("Task completed successfully")
        else:
            # Task failed
            print("Task failed")

# Start consuming messages from each topic
if __name__ == '__main__':
    app.worker_main(argv=['worker', '--loglevel=info'])
    run_parallel_tasks()
