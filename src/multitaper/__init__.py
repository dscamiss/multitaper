"""
The multitaper package computes multitaper spectrum estimates for univariate
time series.  Additionally, the package computes cross-spectrum, coherence,
and transfer function estimates for bivariate time series.
"""

from src.multitaper.mtcross import MTCross, SineCross
from src.multitaper.mtspec import MTSine, MTSpec

__all__ = ["MTCross", "SineCross", "MTSine", "MTSpec"]
__version__ = "1.2.0"
