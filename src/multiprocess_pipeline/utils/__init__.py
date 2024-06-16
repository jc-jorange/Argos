import multiprocessing as mp


def get_current_process_name() -> str:
    return mp.current_process().name
