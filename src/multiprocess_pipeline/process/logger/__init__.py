from enum import Enum, unique

from ._masterclass import *

from .EveryUsage.LogSysUsage import LogSysUsageProcess

@unique
class E_Process_Logger(Enum):
    LogSysUsage = 1


factory_process_logger = {
    E_Process_Logger.LogSysUsage.name: LogSysUsageProcess,
}

__all__ = [
    LogProcess,
    E_Process_Logger,
    factory_process_logger,
]