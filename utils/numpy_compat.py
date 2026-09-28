import warnings

import numpy as np

_LEGACY_ALIASES = {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "str": str,
    "long": int,
    "unicode": str,
}

NumpyCompatNote = "NumPy >= 1.24 removed np.bool/np.int/np.float; medpy 0.4.x still uses them."

def apply_numpy_legacy_aliases():

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, builtin_type in _LEGACY_ALIASES.items():
            try:
                getattr(np, name)
            except AttributeError:
                setattr(np, name, builtin_type)
    return np

apply_numpy_legacy_aliases()
