from celery import Celery, group
from celery.bin import worker

app = Celery('myapp', broker='amqp://guest@localhost//')

@app.task
def consume_topic1():
    # Task logic for topic1

@app.task
def consume_topic2():
    # Task logic for topic2

@app.task
def consume_topic3():
    # Task logic for topic3

def run_parallel_tasks():
    # Create a group of tasks
    task_group = group([consume_topic1.s(), consume_topic2.s(), consume_topic3.s()])

    # Run the tasks in parallel
    result = task_group.apply_async()

    # Wait for the tasks to complete
    result.wait()

    # Check the results of individual tasks
    if result.successful():
        # All tasks were successful
        print("All tasks completed successfully")
    else:
        # At least one task failed
        print("One or more tasks failed")

# Start consuming messages from each topic
if __name__ == '__main__':
    run_parallel_tasks()

    # Start the Celery worker
    worker.worker(app=app).run(
        argv=[
            'worker',
            '--loglevel=info',
            '-Q', 'topic1,topic2,topic3',  # Specify the queues to listen on
            '-c', '1'  # Specify the number of concurrent workers
        ]
    )