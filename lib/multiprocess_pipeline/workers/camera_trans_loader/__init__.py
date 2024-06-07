from enum import Enum, unique
import numpy as np


class BaseCameraTransLoader:
    def __init__(self,
                 source,
                 *args,
                 **kwargs
                 ):
        self.source = source

        self.len = 0
        self.count = 0

    def read_trans(self, idx) -> (int, str, np.ndarray):
        return 0, '', np.ndarray

    def __iter__(self):
        return self

    def __next__(self):
        self.count += 1
        timestamp, path, trans = self.read_trans(self.count)
        if trans:
            return timestamp, path, trans
        else:
            raise StopIteration

    def __getitem__(self, idx):
        idx = idx % len(self)
        return self.read_trans(idx)

    def __len__(self):
        return self.len


from .TransFile import CamTransFileLoader
from .Address import AddressTransLoader
from .Address_All import AddressTransLoader_All


@unique
class E_CameraTransLoaderName(Enum):
    File = 1
    Address = 2
    Address_All = 3


factory_camera_trans_loader = {
    E_CameraTransLoaderName.File.name: CamTransFileLoader,
    E_CameraTransLoaderName.Address.name: AddressTransLoader,
    E_CameraTransLoaderName.Address_All.name: AddressTransLoader_All,
}
