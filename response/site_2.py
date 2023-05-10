import sys
sys.path.append('/Users/nhatcao/multi-topics-video-stream')
from producer.producer import ProducerThread
from config import *
from confluent_kafka import Consumer

site_2_consumer = Consumer(consumer_config)
site_2_consumer.subscribe(["site_2_return"])
print("catching messages")
# Start consuming messages
while True:
    msg = site_2_consumer.poll(1.0)  # Poll for new messages, with a timeout of 1 second

    if msg is None:
        continue

    if msg.error():
        print('Error: {}'.format(msg.error().str()))
        continue

    # Process the received message
    ####update postgres
    print("Committing database")
    cursor = postgres_conn.cursor()
    insert_query = """
    INSERT INTO log_site_2 (log)
    VALUES (%s);
    """
    cursor.execute(insert_query, (str(msg.value().decode('utf-8')),))  # Pass the value as a tuple
    postgres_conn.commit()
    cursor.close()
    # print('Received message: {}'.format(msg.value().decode('utf-8')))

# Close the consumer