from enum import Enum, unique


@unique
class E_pipeline_branch(Enum):
    producer = 1
    consumer = 2
    post = 3
    static_shared_value = 4


from src.multiprocess_pipeline.process.producer import factory_process_producer
from src.multiprocess_pipeline.process.consumer import factory_process_consumer
from src.multiprocess_pipeline.process.post import factory_process_post

factory_process_all = {
    E_pipeline_branch.producer.name: factory_process_producer,
    E_pipeline_branch.consumer.name: factory_process_consumer,
    E_pipeline_branch.post.name: factory_process_post,
}
