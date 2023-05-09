import psycopg2
from pymongo import MongoClient
import gridfs

producer_config = {
    'bootstrap.servers': 'localhost:9092',
    'enable.idempotence': True,
    'acks': 'all',
    'retries': 100,
    'max.in.flight.requests.per.connection': 5,
    'compression.type': 'snappy',
    'linger.ms': 5,
    # 'batch.num.messages': 32,
    'message.max.bytes' : 104858800, 
    # 'replica.fetch.max.bytes': 10485880
    }

consumer_config = {
    'bootstrap.servers': '127.0.0.1:9092',
    'group.id': 'face_recognition',
    'enable.auto.commit': False,
    'max.partition.fetch.bytes': 104858800,
    'auto.offset.reset': 'latest'
    # make consumer reading from start
    # 'default.topic.config': {'auto.offset.reset': 'earliest'}
}

#postgres db
postgres_conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="postgres",
    user="postgres",
    password="123"
)

#mongodb db
mongo_db = MongoClient('mongodb://localhost:27017')["test"]
fs = gridfs.GridFS(mongo_db)

authorization = {'Nhatcao': [True, True, True], 'daniels': [True, False, False], 
                 'Emma': [True, False, False], 'Gupta': [False, True, False], 
                 'Kita': [False, True, False]}

absence_status = {'Nhatcao': False, 'daniels': False, 'Emma': False,
                   'Gupta': False, 'Kita': False}

pickle_encodings = "weight/encodings.pickle"

detection_method = "hog" #cnn