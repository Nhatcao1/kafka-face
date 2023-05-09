import sys
sys.path.append('/Users/nhatcao/multi-topics-video-stream')
from consumer import ConsumerThread
from config import consumer_config

topic_list = ["site_3"]
consumer_faces = ConsumerThread(consumer_config, topic_list, 3)
consumer_faces.read_data()