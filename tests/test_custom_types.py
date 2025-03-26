"""Tests for custom_types.py."""

import numpy as np
import numpy.typing as npt
import pytest

from multitaper.custom_types import MultitaperSetup


@pytest.fixture(name="x")
def fixture_x() -> npt.NDArray:
    """Input vector."""
    return np.random.random(1337)


def test_multitaper_setup_invalid_arg_values(x: npt.NDArray) -> None:
    """Test `MultitaperSetup` with invalid arg types."""
    with pytest.raises(ValueError):
        MultitaperSetup(x=np.random.random((10, 10)))
    with pytest.raises(ValueError):
        MultitaperSetup(x=np.random.random(0))
    with pytest.raises(ValueError):
        MultitaperSetup(x=x, nw=0.0)
    with pytest.raises(ValueError):
        MultitaperSetup(x=x, n_tapers=-1)
    with pytest.raises(ValueError):
        MultitaperSetup(x=x, delta_t=0.0)
    with pytest.raises(ValueError):
        MultitaperSetup(x=x, n_fft=0)
    with pytest.raises(ValueError):
        MultitaperSetup(x=x, n_fft=x.size // 2)
    with pytest.raises(ValueError):
        MultitaperSetup(x=x, weighting_scheme=666)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MultitaperSetup(x=x, water_level=-1.0)
