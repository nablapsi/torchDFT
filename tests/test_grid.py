import math

import torch
from torch.testing import assert_allclose

from torchdft.grid import Uniform1DGrid
from torchdft.utils import gaussian

torch.set_default_dtype(torch.double)


def test_Uniform1DGrid():
    end = 10
    dx = 0.1
    grid = Uniform1DGrid(end=end, dx=dx, reflection_symmetry=False)
    assert grid.grid[0] == 0.0
    assert grid.grid[-1] == end
    assert dx == grid.grid_weights
    assert grid.dv == torch.tensor(1.0)


def test_Uniform1DGrid_integral():
    end = 10
    dx = 0.1
    grid = Uniform1DGrid(end=end, dx=dx, reflection_symmetry=False)
    f = gaussian(grid.grid, mean=5.0, sigma=1.0)
    integral = (f * grid.grid_weights).sum()
    assert_allclose(integral, 1.0)


def test_Uniform1DGrid_reflectionsymmetry():
    end = 10
    dx = 0.1
    grid = Uniform1DGrid(end=end, dx=dx, reflection_symmetry=True)
    assert grid.grid[0] == -end
    assert grid.grid[-1] == end
    assert_allclose(grid.grid, -grid.grid.flip(-1), rtol=0, atol=0)
    assert dx == grid.grid_weights
