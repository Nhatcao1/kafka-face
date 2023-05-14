import sys
sys.path.append('/Users/nhatcao/multi-topics-video-stream')
from consumer import ConsumerThread
from config import consumer_config
from confluent_kafka import Consumer
from multiprocessing import Process, current_process, freeze_support

# Kafka consumer configuration

def start_consumer(topics):
    consumer_faces = ConsumerThread(consumer_config, topics)
    consumer_faces.read_data()

    # print(f"Consumer {current_process().name} subscribed to topics: {topics}")

if __name__ == '__main__':
    freeze_support()
    
    topics = ['site_1', 'site_2', 'site_3']
    num_consumers = 3

    processes = []

    for _ in range(num_consumers):
        process = Process(target=start_consumer, args=(topics,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()