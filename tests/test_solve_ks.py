import torch
from pyscf import dft, gto
from torch.testing import assert_allclose

from torchdft.gaussbasis import GaussianBasis
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


def test_h2_guuss():
    mol = gto.M(atom="H 0 0 0; H 0 0 1.1", basis="cc-pvdz", verbose=3)
    mf = dft.RKS(mol)
    mf.init_guess = "1e"
    mf.xc = "lda,pw"
    energy_true = mf.kernel()
    basis = GaussianBasis(mol)
    density, energy = solve_ks(basis, sum(mol.nelec))
    assert_allclose(energy, energy_true)
