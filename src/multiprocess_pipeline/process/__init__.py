from enum import Enum, unique

from ._masterclass import *

from src.multiprocess_pipeline.process.producer import ProducerProcess, factory_process_producer
from src.multiprocess_pipeline.process.consumer import ConsumerProcess, factory_process_consumer
from src.multiprocess_pipeline.process.post import PostProcess, factory_process_post
from src.multiprocess_pipeline.process.logger import LogProcess, factory_process_logger


factory_process_all = {
    E_pipeline_branch.producer.name: factory_process_producer,
    E_pipeline_branch.consumer.name: factory_process_consumer,
    E_pipeline_branch.post.name: factory_process_post,
}
