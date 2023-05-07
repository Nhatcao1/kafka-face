from confluent_kafka.admin import AdminClient, NewTopic

class createTopic:
    def __init__(self, n_replicas, n_partitions, admin_client):
        self.n_replicas = n_replicas
        self.n_partitions = n_partitions
        self.admin_client = admin_client
        self.topics = []

    def add_topic(self, name):
        self.topics.append(NewTopic(name, self.n_partitions, self.n_replicas))
        
    def create_topic(self):
        self.admin_client.create_topics(self.topics)

    def check_info_topic(self):
        topic_info = self.admin_client.list_topics().topics
        # print(topic_info)
        for item in topic_info:
            print(item)


# create/assign topic according to name
##################
#create topics
topics = createTopic(1,3,AdminClient({
    "bootstrap.servers": "localhost:9092"
}))

topics.add_topic("site_1")
topics.add_topic("site_2")
topics.add_topic("site_3")

#trigger
topics.add_topic("site_1_return")
topics.add_topic("site_2_return")
topics.add_topic("site_3_return")

topics.create_topic()
topics.check_info_topic()


# n_repicas = 1
# n_partitions = 3