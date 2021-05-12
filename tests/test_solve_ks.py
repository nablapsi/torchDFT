import torch
from torch.testing import assert_allclose

from torchdft.gridbasis import GridBasis
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
    )
    grid = torch.arange(-10, 10, 0.1)
    basis = GridBasis(H2, grid)
    density, energy = solve_ks(basis, H2.nelectrons)
    assert_allclose(energy, -1.4045913)
