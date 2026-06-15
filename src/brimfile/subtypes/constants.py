from enum import Enum
from dataclasses import dataclass

__docformat__ = "google"

class SubType(Enum):
    """
    Enumeration of supported Brimfile subtypes.
    See https://github.com/brillouin-imaging/Brillouin-standard-file/blob/main/docs/brim_file_subtypes.md for their definition.
    """
    none = 'none'
    SinglePoint_VIPA_v0_1 = 'SinglePoint_VIPA_v0.1'

@dataclass
class Feature:
    """
    Class representing a feature of a Brimfile subtype, which can be required or optional.
    """
    name: str
    required: bool
    description: str | None = None


FEATURES = {
    SubType.SinglePoint_VIPA_v0_1: [
        Feature(
            name='2DArray_per_spectrum',
            required=True
        ),
        Feature(
            name='Spectral_line',
            required=False
        )
    ]
}