from producer import ProducerThread
from config import producer_config, topic_name

site_1 = ProducerThread(producer_config, "site_1", "videos/daniels.mov")
site_1.start()