from .producer import E_Global_Process_Producer, factory_global_process_producer
from .consumer import E_Global_Process_Consumer, factory_global_process_consumer
from .post import E_Global_Process_Post, factory_global_process_post

from lib.multiprocess_pipeline.pipeline_config import E_pipeline_station

factory_global_process_all = {
    E_pipeline_station.producer: factory_global_process_producer,
    E_pipeline_station.consumer: factory_global_process_consumer,
    E_pipeline_station.post: factory_global_process_post
}
