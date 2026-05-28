"""
Matplotlib fallback helpers for SVG export when Plotly/Kaleido is unavailable.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Tuple


def init_matplotlib() -> Tuple[object, object]:
    """Initialize matplotlib in headless mode and return (matplotlib, pyplot)."""
    # Ensure a writable cache directory to avoid runtime warnings.
    cache_root = Path(tempfile.gettempdir()) / "xdg-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    mpl_cache = cache_root / "matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    import matplotlib

    # Use a non-interactive backend for headless environments.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433 (runtime import)

    return matplotlib, plt


def rgb_to_mpl(color: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert 0-255 RGB tuple to 0-1 matplotlib tuple."""
    return (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
