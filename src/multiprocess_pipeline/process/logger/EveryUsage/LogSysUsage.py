from .._masterclass import LogProcess
from src.utils.logger import ALL_LoggerContainer

from typing import Dict
import psutil
from pynvml import *


class LogSysUsageProcess(LogProcess):
    prefix = 'Argos-SubProcess-LogSysUsageProcess_'
    dir_name = 'sys_usage_log'
    log_name = 'Global_SysUsage_Log'

    def __init__(self,
                 process_dict: dict,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs, output_to_screen=False)
        self.process_dict: Dict[str: int]
        self.process_dict = process_dict
        self.logger_process_dict = {}

    def run_begin(self) -> None:
        super(LogSysUsageProcess, self).run_begin()
        for each_process_name in self.process_dict.keys():
            each_name = self.name + "_" + each_process_name
            each_logger = ALL_LoggerContainer.add_logger(each_name)
            log_level = 'debug' if self.opt.debug else 'info'
            ALL_LoggerContainer.set_logger_level(each_name, log_level)

            ALL_LoggerContainer.add_file_handler(each_name, each_process_name, self.main_save_dir)

            self.logger.info(f'create system usage log for {each_process_name}')
            self.logger_process_dict.update({each_process_name: each_logger})

    def log_cpu_initial(self):
        logger = self.logger

        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        max_cpu_freq = cpu_freq.max

        sys_memory_info = psutil.virtual_memory()
        total_memory = sys_memory_info.total / (1024**2)

        logger.info(f"CPU count: {cpu_count}, "
                    f"max frequency: {max_cpu_freq} Mhz, "
                    f"total memory: {total_memory} MB")

    def log_cpu_total(self):
        logger = self.logger

        cpu_count = psutil.cpu_count()

        cpu_freq = psutil.cpu_freq()
        cur_cpu_freq = cpu_freq.current

        avg_load = psutil.getloadavg()
        total_load = [x / cpu_count * 100 for x in avg_load]

        sys_memory_info = psutil.virtual_memory()
        total_memory = sys_memory_info.total
        used_memory = sys_memory_info.used
        available_memory = sys_memory_info.available

        logger.info(f"CPU "
                    f"current frequency: {cur_cpu_freq} Mhz, "
                    f"total load: {round(total_load[0],4)}% in past 1 minute, "
                    f"used memory: {used_memory / (1024**2)} MB {round(used_memory / total_memory, 6)*100}%, "
                    f"available memory: {available_memory / (1024**2)} MB {round(available_memory / total_memory, 6)*100}%")

    def log_cpu_process(self, pname: str, pid) -> None:
        process = psutil.Process(pid)
        logger = self.logger_process_dict[pname]

        cpu_usage = process.cpu_percent()

        memory_info = process.memory_info()
        memory_usage = memory_info.rss / (1024**2)  # in MB
        virtual_memory_usage = memory_info.vms / (1024**2)  # in MB
        memory_usage_rate = process.memory_percent()

        logger.info(f"CPU: "
                    f"CPU cores usage: {cpu_usage} %, "
                    f"physic memory usage: {memory_usage} MB, "
                    f"virtual memory usage: {virtual_memory_usage} MB, "
                    f"physic memory usage rate: {round(memory_usage_rate, 4)}%")

    def log_gpu_initial(self):
        nvmlInit()  # initialize
        logger = self.logger

        gpuDeriveInfo = nvmlSystemGetDriverVersion()
        logger.info(f"Driver version: {gpuDeriveInfo} ")  # Driver info

        gpuDeviceCount = nvmlDeviceGetCount()  # Nvidia GPU count
        logger.info(f"GPU count：{gpuDeviceCount}")

        for i in range(gpuDeviceCount):
            handle = nvmlDeviceGetHandleByIndex(i)  # get handle of GPU i

            memoryInfo = nvmlDeviceGetMemoryInfo(handle)  # get Gpu i memory info

            gpuName = nvmlDeviceGetName(handle)

            logger.info(f"GPU #{i} Name: {gpuName}, Total Memory: {memoryInfo.total / (1024**2)} MB")

    def log_gpu_total(self):
        logger = self.logger

        gpuDeviceCount = nvmlDeviceGetCount()  # Nvidia GPU count

        for i in range(gpuDeviceCount):
            handle = nvmlDeviceGetHandleByIndex(i)  # get handle of GPU i

            memoryInfo = nvmlDeviceGetMemoryInfo(handle)  # get Gpu i memory info

            gpuTemperature = nvmlDeviceGetTemperature(handle, 0)

            gpuFanSpeed = nvmlDeviceGetFanSpeed(handle)

            gpuPerformanceState = nvmlDeviceGetPerformanceState(handle)
            gpuPowerUsage = nvmlDeviceGetPowerUsage(handle)

            gpuUtilRate = nvmlDeviceGetUtilizationRates(handle).gpu
            gpuMemoryRate = nvmlDeviceGetUtilizationRates(handle).memory

            logger.info(f"GPU #{i} "
                        f"Total core usage rate: {gpuUtilRate} %, "
                        f"Total Memory usage rate: {round(memoryInfo.used / memoryInfo.total, 4)*100} %, "
                        f"Total Memory WriteRead rate: {gpuMemoryRate}, "
                        f"Total Memory used: {memoryInfo.used / (1024**2)} MB, "
                        f"Total Memory free: {memoryInfo.free / (1024 ** 2)} MB, "
                        f"Performance State: {gpuPerformanceState}, "
                        f"Power usage: {gpuPowerUsage / 1000} W, "
                        f"Temperature: {gpuTemperature} C, "
                        f"Fan Speed: {gpuFanSpeed} %")

    def log_gpu_process(self, pname: str, pid):
        logger = self.logger_process_dict[pname]

        gpuDeviceCount = nvmlDeviceGetCount()

        for i in range(gpuDeviceCount):
            handle = nvmlDeviceGetHandleByIndex(i)

            memoryInfo = nvmlDeviceGetMemoryInfo(handle)

            # for each process in this GPU
            pidAllInfo = nvmlDeviceGetComputeRunningProcesses(handle)
            for pidInfo in pidAllInfo:
                if pidInfo.pid == pid and pidInfo.usedGpuMemory:
                    logger.info(f"GPU #{i} in Process {pname} "
                                f"Memory usage rate: {round(pidInfo.usedGpuMemory / memoryInfo.total, 6)*100} %, "
                                f"Memory used: {pidInfo.usedGpuMemory / (1024 ** 2)} MB")

    def run_action(self) -> None:
        super().run_action()
        self.logger.info('Start logging')

        bHaveProcessAlive = True
        self.log_cpu_initial()
        self.log_gpu_initial()

        while bHaveProcessAlive:
            try:
                bHaveProcessAlive = False

                self.log_cpu_total()
                self.log_gpu_total()

                for process_name, process_id in self.process_dict.items():
                    bThisProcessAlive = psutil.pid_exists(process_id)
                    bHaveProcessAlive = bHaveProcessAlive or bThisProcessAlive
                    if bThisProcessAlive:
                        self.log_cpu_process(process_name, process_id)
                        self.log_gpu_process(process_name, process_id)
            except psutil.NoSuchProcess:
                continue

    def run_end(self) -> None:
        super(LogSysUsageProcess, self).run_end()
        nvmlShutdown()
