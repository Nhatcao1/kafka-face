from consumer import ConsumerThread
from config import consumer_config

topic_list = ["site_1", "site_2"]
# topic_list = ["site_1"]
consumer_faces = ConsumerThread(consumer_config, topic_list)
consumer_faces.get_name_from_database()
# consumer_faces.read_data()
# consumer_faces.update_database()
consumer_faces.start(3)