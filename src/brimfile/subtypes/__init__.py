"""
Utilities for working with brimfile subtypes.

This module exposes the subtype enum and helper used to inspect the subtype
declared at the root of a brim file:

- ``SubType``: enum of supported subtype identifiers.
- ``get_subtype``: read the subtype from a file abstraction.

Subtype-specific helpers are implemented in dedicated modules under
``brimfile.subtypes`` (for example ``single_point_VIPA``).

Quick usage
-----------

When a file is opened, you can inspect its subtype directly from ``File``:

```Python
	import brimfile as brim

	f = brim.File("path/to/your/file.brim.zarr")
	subtype = f.subtype
	print(subtype)
	f.close()
```

Subtype-specific APIs are available from their corresponding module. For
``SinglePoint_VIPA_v0.1``, for example:

```Python
	from brimfile.subtypes import single_point_VIPA

	single_point_VIPA.add_rawdata(data_group, raw_data)
	raw_spectrum, spectral_line, linewidth = single_point_VIPA.get_raw_spectrum_in_image(
		data_group,
		(z, y, x)
	)
```
"""

from .constants import SubType
from .utils import get_subtype