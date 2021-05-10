import torch
from torch.testing import assert_allclose

from torchdft.scf import solve_ks
from torchdft.utils import System


def test_h2():
    charges = torch.tensor([1.0, 1.0])
    centers = torch.tensor([0.0, 1.401118437])
    nelectrons = 2
    H2 = System(
        charges=charges,
        centers=centers,
        nelectrons=nelectrons,
        vext=None,
        density=None,
        energy=None,
    )
    grid = torch.arange(-10, 10, 0.1)
    system = solve_ks(H2, grid)
    assert_allclose(system.energy, -2.0)