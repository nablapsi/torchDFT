import math

import torch
from torch.testing import assert_allclose

from torchdft.density import Density
from torchdft.grid import Uniform1DGrid
from torchdft.gridbasis import GridBasis
from torchdft.kinetic_functionals import ThomasFermi1D, VonWeizsaecker
from torchdft.utils import System, gaussian


class TestKineticFunctionals:
    def test_get_TF_energy_1d(self):
        grid = torch.arange(-10, 10, 0.1)
        density = Density(gaussian(grid, 0, 1), grid, grid)

        assert_allclose(
            1 / (2.0 * math.sqrt(math.pi)),
            (ThomasFermi1D(c=1.0)(density) / density.value).sum() * 0.1,
        )

    def test_get_vW_energy(self):
        grid = Uniform1DGrid(end=10, dx=0.1, reflection_symmetry=True)
        system = System(
            centers=torch.tensor([0]),
            Z=torch.tensor([1]),
        )
        basis = GridBasis(system, grid)
        density = Density(gaussian(grid.nodes, 0, 1), grid.nodes, grid.grid_weights)
        density.grad = basis.get_density_gradient(
            (density.value * basis.grid_weights).diag_embed()
        )

        assert_allclose(1.0 / 8.0, (VonWeizsaecker()(density)).sum() * 0.1)
