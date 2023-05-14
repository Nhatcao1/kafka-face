import sys
sys.path.append('/Users/nhatcao/multi-topics-video-stream')
from multiprocessing import Process, freeze_support
from confluent_kafka import KafkaException, Consumer
from config import *

def consume_messages(topic, site_number):
    consumer = Consumer(consumer_config)
    consumer.subscribe([topic])
    print(f"Subscribed to topic: {topic}")

    try:
        while True:
            msg = consumer.poll(1.0)

            if msg is None:
                continue

            if msg.error():
                print(f"Error: {msg.error().str()}")
                continue

            # Process the received message
            log = msg.value().decode('utf-8')
            print(f"Received message from topic {topic}: {log}")
            site = "log_site_" + str(site_number)
            # Insert log into PostgreSQL
            cursor = postgres_conn.cursor()
            insert_query = f"INSERT INTO {site} (log) VALUES (%s);"
            cursor.execute(insert_query, (log,))
            postgres_conn.commit()
            cursor.close()

    except KeyboardInterrupt:
        consumer.close()


if __name__ == '__main__':
    freeze_support()

    # Create and start consumer processes
    consumer_processes = [
        Process(target=consume_messages, args=("site_1_return", 1)),
        Process(target=consume_messages, args=("site_2_return", 2)),
        Process(target=consume_messages, args=("site_3_return", 3))
    ]

    for consumer_process in consumer_processes:
        consumer_process.start()

    # Wait for all processes to finish
    for consumer_process in consumer_processes:
        consumer_process.join()