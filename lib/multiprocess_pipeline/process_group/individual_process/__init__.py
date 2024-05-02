from .producer import E_Indi_Process_Producer, factory_indi_process_producer
from .consumer import E_Indi_Process_Consumer, factory_indi_process_consumer
from .post import E_Indi_Process_Post, factory_indi_process_post

from lib.multiprocess_pipeline.pipeline_config import E_pipeline_station

factory_indi_process_all = {
    E_pipeline_station.producer: factory_indi_process_producer,
    E_pipeline_station.consumer: factory_indi_process_consumer,
    E_pipeline_station.post: factory_indi_process_post
}
