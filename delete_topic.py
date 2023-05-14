from confluent_kafka.admin import AdminClient, NewTopic

# Kafka broker configuration
bootstrap_servers = "localhost:9092"
admin_config = {"bootstrap.servers": bootstrap_servers}

# Topic name to delete
topic_name = "site_1"

def delete_topic(topic_name):
    # Create an instance of the AdminClient
    admin_client = AdminClient(admin_config)

    # Create a list of topics to delete
    topics_to_delete = ["site_1","site_2","site_3"]

    # # Create the NewTopic objects for deletion
    # topics = [NewTopic(topic, num_partitions=3, replication_factor=1) for topic in topics_to_delete]

    # Delete the topics
    deleted_topics = admin_client.delete_topics(topics_to_delete)

    # Wait for topic deletion to complete
    for topic, deletion_result in deleted_topics.items():
        try:
            deletion_result.result()  # Wait for the deletion to complete
            print(f"Topic '{topic}' deleted successfully.")
        except Exception as e:
            print(f"Failed to delete topic '{topic}': {e}")

# Call the delete_topic function
delete_topic(topic_name)