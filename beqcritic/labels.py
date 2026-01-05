from __future__ import annotations

from typing import Any
import math

import numpy as np

_TRUE_STRINGS = {"1", "true", "yes", "correct", "ok"}
_FALSE_STRINGS = {"0", "false", "no", "incorrect", "wrong"}


def coerce_binary_label(x: Any) -> int:
    if isinstance(x, (bool, np.bool_)):
        return int(x)
    if isinstance(x, (int, np.integer)):
        val = int(x)
        if val in (0, 1):
            return val
        raise ValueError(f"Label must be 0 or 1, got {x!r}")
    if isinstance(x, (float, np.floating)):
        fval = float(x)
        if math.isfinite(fval) and fval.is_integer():
            val = int(fval)
            if val in (0, 1):
                return val
        raise ValueError(f"Label must be 0 or 1, got {x!r}")
    if isinstance(x, str):
        xl = x.strip().lower()
        if xl in _TRUE_STRINGS:
            return 1
        if xl in _FALSE_STRINGS:
            return 0
        try:
            fval = float(xl)
        except ValueError:
            raise ValueError(f"Cannot interpret label value: {x!r}") from None
        if math.isfinite(fval) and fval.is_integer():
            val = int(fval)
            if val in (0, 1):
                return val
        raise ValueError(f"Label must be 0 or 1, got {x!r}")
    raise ValueError(f"Cannot interpret label value: {x!r}")
