import math

import torch
from torch.testing import assert_allclose

from torchdft.density import Density
from torchdft.gridbasis import get_gradient
from torchdft.kinetic_functionals import ThomasFermi1D, VonWeizsaecker
from torchdft.utils import gaussian, get_dx


class TestKineticFunctionals:
    def test_get_TF_energy_1d(self):
        grid = torch.arange(-10, 10, 0.1)
        density = Density(gaussian(grid, 0, 1))

        assert_allclose(
            1 / (2.0 * math.sqrt(math.pi)), (ThomasFermi1D(c=1.0)(density) / density.value).sum() * 0.1
        )

    def test_get_vW_energy(self):
        grid = torch.arange(-10, 10, 0.1)
        dx = get_dx(grid)
        density = Density(gaussian(grid, 0, 1))
        grad_operator = get_gradient(grid) / dx
        density.grad = grad_operator.mv(density.value)

        assert_allclose(
            1.0 / 8.0, (VonWeizsaecker()(density)).sum() * 0.1
        )
