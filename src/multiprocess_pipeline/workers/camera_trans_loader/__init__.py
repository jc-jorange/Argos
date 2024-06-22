from enum import Enum, unique

from .file import CamTransFileLoader
from .address import AddressTransLoader
from .address_all import AddressTransLoader_All


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
