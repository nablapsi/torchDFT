import torch
from pyscf import dft, gto
from torch.testing import assert_allclose

from torchdft.gaussbasis import GaussianBasis
from torchdft.gridbasis import GridBasis
from torchdft.scf import solve_scf
from torchdft.utils import System
from torchdft.xc_functionals import Lda1d, LdaPw92


def test_h2():
    charges = torch.tensor([1.0, 1.0])
    centers = torch.tensor([0.0, 1.401118437])
    nelectrons = 2
    H2 = System(charges=charges, centers=centers, nelectrons=nelectrons)
    grid = torch.arange(-10, 10, 0.1)
    basis = GridBasis(H2, grid)
    density, energy = solve_scf(basis, H2.nelectrons, Lda1d())
    assert_allclose(energy, -1.4045913)


def test_ks_of():
    charges = torch.tensor([1.0, 1.0])
    centers = torch.tensor([0.0, 1.401118437])
    nelectrons = 2
    H2 = System(charges=charges, centers=centers, nelectrons=nelectrons)
    grid = torch.arange(-10, 10, 0.1)
    basis = GridBasis(H2, grid)
    density_ks, energy_ks = solve_scf(basis, H2.nelectrons, Lda1d())
    density_of, energy_of = solve_scf(basis, H2.nelectrons, Lda1d(), mode="OF")
    assert_allclose(density_ks, density_of)
    assert_allclose(energy_ks, energy_of)


def test_h2_guuss():
    mol = gto.M(atom="H 0 0 0; H 0 0 1.1", basis="cc-pvdz", verbose=3)
    mf = dft.RKS(mol)
    mf.init_guess = "1e"
    mf.xc = "lda,pw"
    energy_true = mf.kernel()
    basis = GaussianBasis(mol)
    density, energy = solve_scf(basis, sum(mol.nelec), LdaPw92())
    assert_allclose(energy, energy_true)
