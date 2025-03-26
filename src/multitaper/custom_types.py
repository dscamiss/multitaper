"""Custom type definitions."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy.typing as npt


class WeightingScheme(IntEnum):
    """Enum for multitaper weighting schemes."""

    ADAPTIVE = 1
    UNITY = 2
    EIGENVALUE = 3


@dataclass
class MultitaperSetup:
    """
    Dataclass for multitaper setup parameters.

    Args:
        x: Input vector of length `n`.
        nw: Time-bandwidth product. Defaults to 4.0.
        n_tapers: Number of tapers. Defaults to `round(2 * nw - 1)`.
        delta_t: Input sampling interval. Defaults to 1.0.
        n_fft: Number of FFT points. Defaults to `2 * n + 1`.
        weighting_scheme: Weighting scheme. Defaults to adaptive weighting.
        water_level: Water level for deconvolution. Defaults to 0.0.

    Raises:
        ValueError: If any arguments are invalid.
    """

    x: npt.NDArray
    nw: float = 4.0
    n_tapers: Optional[int] = None
    delta_t: float = 1.0
    n_fft: Optional[int] = None
    weighting_scheme: WeightingScheme = WeightingScheme.ADAPTIVE
    water_level: float = 0.0

    def _set_default_values(self) -> None:
        if self.n_tapers is None:
            self.n_tapers = int(round(2 * self.nw - 1))
        if self.n_fft is None:
            self.n_fft = 2 * self.x.size + 1

    def _check_arg_values(self) -> None:
        if self.x.ndim != 1:
            raise ValueError(f"Input vector has ndim = {self.x.ndim}")
        if self.x.size == 0:
            raise ValueError("Input vector is empty")
        if self.nw <= 0.0:
            raise ValueError("Time-bandwidth product must be positive")
        if self.n_tapers < 1:  # type: ignore
            raise ValueError("Number of tapers must be positive")
        if self.delta_t <= 0.0:
            raise ValueError("Sampling interval must be positive")
        if self.n_fft <= 0:  # type: ignore
            raise ValueError("Number of FFT points must be positive")
        if self.x.size > self.n_fft:  # type: ignore
            raise ValueError("Input vector size exceeds number of FFT points")
        if self.weighting_scheme not in iter(WeightingScheme):
            raise ValueError("Invalid weighting scheme")
        if self.water_level < 0.0:
            raise ValueError("Water level must be non-negative")

    def __post_init__(self) -> None:
        self._set_default_values()
        self._check_arg_values()
