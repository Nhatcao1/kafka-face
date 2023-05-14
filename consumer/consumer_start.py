import sys
sys.path.append('/Users/nhatcao/multi-topics-video-stream')
from celery import Celery, group
from celery.bin import worker
from confluent_kafka import Consumer, Producer
from datetime import datetime
import face_recognition
from config import *
import pickle
import cv2
import numpy as np


# Kafka broker configuration
bootstrap_servers = 'localhost:9092'
group_id = 'my-consumer-group'

# Create Kafka consumer configuration
consumer_config = {
    'bootstrap.servers': bootstrap_servers,
    'group.id': group_id,
    'auto.offset.reset': 'earliest'
}

# Create Celery application
app = Celery('kafka_consumer', broker='pyamqp://guest@localhost//')

# Create Kafka consumer
consumer = Consumer(consumer_config)

# Task to process a Kafka message
@app.task
def process_message(msg):
    # Process the received message
    print('Received message: {}'.format(msg.value().decode('utf-8')))

# Celery worker entry point
def run_worker():
    try:
        while True:
            msg = consumer.poll(1.0)  # Poll for new messages
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition, continue polling
                    continue
                else:
                    # Handle error
                    raise KafkaException(msg.error())
            # Process the received message asynchronously using Celery
            process_message.delay(msg)
    except KeyboardInterrupt:
        # Stop consuming and close the consumer
        consumer.close()

if __name__ == '__main__':
    app.worker_main(argv=['worker', '--loglevel=info'])
    # run_worker()