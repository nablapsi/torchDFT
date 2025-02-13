import torch
from torch.testing import assert_allclose

from torchdft.grid import RadialGrid
from torchdft.radialbasis import RadialBasis
from torchdft.utils import System, gaussian


class TestFunctionals:
    def test_get_gradient(self):
        mean, std = 0, 1
        grid = RadialGrid(end=5, dx=1e-2)
        system = System(centers=torch.tensor([0]), Z=torch.tensor([1]))
        basis = RadialBasis(system, grid)
        density = gaussian(grid.nodes, mean, std)
        den_der = basis.get_density_gradient(
            (density * basis.dv * basis.grid_weights[..., None]).diag_embed()
        )
        assert_allclose(
            den_der, -(grid.nodes - mean) / std * density, atol=1e-5, rtol=1e-4
        )
