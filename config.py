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

topic_name = {
    
}