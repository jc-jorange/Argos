import torch


def check_batch_size(device: torch.device, batch_size: int = 0) -> None:
    if device != torch.device('cpu'):
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, f'batch-size {batch_size} not multiple of GPU count {ng}'
