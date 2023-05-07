from producer import ProducerThread
from config import producer_config

site_3 = ProducerThread(producer_config, "site_3", 0)
site_3.start()