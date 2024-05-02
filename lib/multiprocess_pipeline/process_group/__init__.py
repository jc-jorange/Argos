from .global_process import factory_global_process_all
from .individual_process import factory_indi_process_all

from ..pipeline_config import E_pipeline_group

factory_process_all = {
    E_pipeline_group.indis: factory_indi_process_all,
    E_pipeline_group.globals: factory_global_process_all
}
