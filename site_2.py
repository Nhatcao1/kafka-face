from producer import ProducerThread
from config import producer_config

site_2 = ProducerThread(producer_config, "site_2", "videos/Emma.mov")
site_2.start()