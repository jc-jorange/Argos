from enum import Enum, unique

from .address import AddressDataSender


@unique
class E_DataSenderName(Enum):
    File = 1
    Address = 2


factory_data_sender = {
    E_DataSenderName.File.name: None,
    E_DataSenderName.Address.name: AddressDataSender,
}