import sys
sys.path.append('/Users/nhatcao/multi-topics-video-stream')
from config import consumer_config
from confluent_kafka import Consumer

site_1_consumer = Consumer(consumer_config)
site_1_consumer.subscribe(["site_1_return"])

# Start consuming messages
while True:
    msg = site_1_consumer.poll(1.0)  # Poll for new messages, with a timeout of 1 second

    if msg is None:
        continue

    if msg.error():
        print('Error: {}'.format(msg.error().str()))
        continue

    # Process the received message
    print('Received message: {}'.format(msg.value().decode('utf-8')))

# Close the consumer