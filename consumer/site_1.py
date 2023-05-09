import sys
sys.path.append('/Users/nhatcao/multi-topics-video-stream')
from consumer import ConsumerThread
from config import consumer_config

topic_list = ["site_1"]
consumer_faces = ConsumerThread(consumer_config, topic_list, 1)
consumer_faces.read_data()