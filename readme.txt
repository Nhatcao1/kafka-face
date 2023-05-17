# Kafka Producer and Consumer Setup

This repository provides scripts and instructions for setting up a Kafka producer and consumer using Docker containers.

## Prerequisites

- Docker
- Docker Compose

## Kafka Setup

1. Clone this repository.
2. Start the Kafka and ZooKeeper containers using `docker-compose up -d`.

## Running the Scripts

### Step 1: Create Topics

Run `python create_topic.py` to create Kafka topics.

### Step 2: Run the Consumer

Navigate to the `consumer` directory and execute `python parallel_consumer.py` to run the consumer.

### Step 3: Run the Response Script

Navigate to the `response` directory and execute `python response.py` to run the response script.

### Step 4: Run the Producer

Navigate to the `producer` directory and execute `python producer_thread.py` to run the producer.

## Configuration

Configure the scripts based on your requirements.

## Docker Images

- Kafka: `wurstmeister/kafka`
- ZooKeeper: `wurstmeister/zookeeper`

Feel free to modify the code and configurations as needed.
