import os


def model_dir_name(module_file: str):
    return os.path.split(os.path.dirname(module_file))[1]
