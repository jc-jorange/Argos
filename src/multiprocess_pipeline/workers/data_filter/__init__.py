from enum import Enum, unique

from .specialID import SpecialIDDataFilter
from .first_valid import FirstValidDataFilter


@unique
class E_DataFilterName(Enum):
    SpecialID = 1
    FirstValid = 2


factory_data_filter = {
    E_DataFilterName.SpecialID.name: SpecialIDDataFilter,
    E_DataFilterName.FirstValid.name: FirstValidDataFilter,
}
