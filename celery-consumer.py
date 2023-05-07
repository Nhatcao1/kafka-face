from celery import Celery
from confluent_kafka import Consumer, Producer
import psycopg2
from datetime import datetime
import face_recognition
from config import consumer_config,producer_config
import pickle
import cv2
import numpy as np

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="postgres",
    user="postgres",
    password="123"
)

pickle_encodings = "weight/encodings.pickle"
authorization = {}

def get_authorization():
    cur = conn.cursor()
    query = "SELECT * FROM your_table"
    cur.execute(query)
    # Fetch all rows from the result
    rows = cur.fetchall()
    for row in rows:
        # name, site1, ....
        authorization[row[0]] = [row[1], row[2], row[3]]
    cur.close()



# Create a Celery app
app = Celery('multi_consumer', broker='amqp://guest@localhost//')

def SingleShotProducer(name, topic, site):
    producer = Producer(producer_config)
    message = "UnKnown person"
    if authorization[name][site - 1]:
        message = "Employee " + name + "is authorized to site: " + str(site)
        print(message)
    
    # the topic for return channel have _return
    producer.produce(topic + "_return", value = message)
    producer.flush()
    producer.close()

# Task to update PostgreSQL
# Function to log entrance event
def log_entrance_event(employee_name, time_at_entrance, site):
    cursor = conn.cursor()
    # timestamp = datetime.now()
    insert_query = """
    INSERT INTO entrance_logs (employee_name, time_at_entrance, site)
    VALUES (%s, %s, %s);
    """
    cursor.execute(insert_query, (employee_name, time_at_entrance, site))
    conn.commit()
    cursor.close()
    print("Entrance event logged successfully!")

def log_absence(employee_name, absense_status=True):
    cursor = conn.cursor()
    update_query = """
    UPDATE absense
    SET absense_status = %s
    WHERE employee_name = %s;
    """
    cursor.execute(update_query, (employee_name, absense_status))
    conn.commit()
    cursor.close()
    print("Absense status updated successfully!")

def process_message(message, topic, site):
    # process the message
    nparr = np.frombuffer(message.value(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    data = pickle.loads(open(pickle_encodings, "rb").read())
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb,model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)
    name = "Unknown"
    
    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            # determine the recognized face with the largest number of
            # votes (note: in the event of an unlikely tie Python will
            # select first entry in the dictionary)
            name = max(counts, key=counts.get)
            #supposed that every turn, only one face
    
    SingleShotProducer(name, topic, site)
    timestamp = datetime.now()
    log_entrance_event(employee_name=name, time_at_entrance=timestamp, site = site)
    log_absence(name)

####
# Define tasks for each topic
@app.task
def consume_topic1():
    consumer = Consumer(consumer_config)
    consumer.subscribe(["site_1"])

    for message in consumer:
        process_message.delay(message, "site_1", 1)

@app.task
def consume_topic2():
    consumer = Consumer(consumer_config)
    consumer.subscribe(["site_2"])

    for message in consumer:
        process_message.delay(message, "site_2", 2)

@app.task
def consume_topic3():
    consumer = Consumer(consumer_config)
    consumer.subscribe(["site_3"])

    for message in consumer:
        process_message.delay(message, "site_3", 3)

# Start consuming messages from each topic
if __name__ == '__main__':
    # app.worker_main(['multi_consumer', '-Q', 'topic1', '-c', '1', '-n', 'topic1_worker'])
    # app.worker_main(['multi_consumer', '-Q', 'topic2', '-c', '1', '-n', 'topic2_worker'])
    # app.worker_main(['multi_consumer', '-Q', 'topic3', '-c', '1', '-n', 'topic3_worker'])
    app.worker_main(['multi_consumer', '--loglevel=info'])

    #celery -A <your_module_name> worker --loglevel=info

    # export PATH="/path/to/celery:$PATH"
