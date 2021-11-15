import torch
from torch.testing import assert_allclose

from torchdft.radialbasis import RadialBasis
from torchdft.utils import System, gaussian


class TestFunctionals:
    def test_get_gradient(self):
        mean, std = 0, 1
        grid = torch.linspace(1, 5, 100, dtype=torch.double)
        system = System(centers=torch.tensor([0]), Z=torch.tensor([1]), grid=grid)
        basis = RadialBasis(system)
        density = gaussian(grid, mean, std)
        den_der = basis._get_density_gradient(density)
        assert_allclose(den_der, -(grid - mean) / std * density, atol=1e-5, rtol=1e-4)
