import math

import torch
from torch.testing import assert_allclose

from torchdft.grid import RadialGrid, Uniform1DGrid
from torchdft.utils import gaussian

torch.set_default_dtype(torch.double)


def test_Uniform1DGrid():
    end = 10
    dx = 0.1
    grid = Uniform1DGrid(end=end, dx=dx, reflection_symmetry=False)
    assert grid.nodes[0] == 0.0
    assert grid.nodes[-1] == end
    assert dx == grid.grid_weights
    assert grid.dv == torch.tensor(1.0)


def test_Uniform1DGrid_integral():
    end = 10
    dx = 0.1
    grid = Uniform1DGrid(end=end, dx=dx, reflection_symmetry=False)
    f = gaussian(grid.nodes, mean=5.0, sigma=1.0)
    integral = (f * grid.grid_weights).sum()
    assert_allclose(integral, 1.0)


def test_Uniform1DGrid_reflectionsymmetry():
    end = 10
    dx = 0.1
    grid = Uniform1DGrid(end=end, dx=dx, reflection_symmetry=True)
    assert grid.nodes[0] == -end
    assert grid.nodes[-1] == end
    assert_allclose(grid.nodes, -grid.nodes.flip(-1), rtol=0, atol=0)
    assert dx == grid.grid_weights


def test_RadialGrid():
    end = 10
    dx = 0.1
    grid = RadialGrid(end=end, dx=dx)
    assert grid.nodes[-1] == end
    assert dx == grid.grid_weights
    assert grid.grid_weights == dx
    assert_allclose(
        grid.grid_weights * grid.dv, 4 * math.pi * grid.nodes**2 * dx, rtol=0, atol=0
    )


def test_RadialGrid_integral():
    end = 20
    dx = 0.01
    grid = RadialGrid(end=end, dx=dx)
    f = (-grid.nodes).exp()
    integral = (f * grid.grid_weights * grid.dv).sum()
    assert_allclose(integral, 4 * math.pi * 2)
