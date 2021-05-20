import torch
from pyscf import dft, gto
from torch.testing import assert_allclose

from torchdft.gaussbasis import GaussianBasis
from torchdft.gridbasis import BatchGridBasis, GridBasis
from torchdft.scf import ks_iteration, solve_scf
from torchdft.utils import GeneralizedDiagonalizer, System, SystemBatch
from torchdft.xc_functionals import PBE, Lda1d, LdaPw92


def test_h2():
    charges = torch.tensor([1.0, 1.0])
    centers = torch.tensor([0.0, 1.401118437])
    nelectrons = 2
    H2 = System(nelectrons, charges, centers)
    grid = torch.arange(-10, 10, 0.1)
    basis = GridBasis(H2, grid)
    density, energy = solve_scf(basis, H2.occ(), Lda1d())
    assert_allclose(energy, -1.4045913)


def test_ks_of():
    charges = torch.tensor([1.0, 1.0])
    centers = torch.tensor([0.0, 1.401118437])
    nelectrons = 2
    H2 = System(nelectrons, charges, centers)
    grid = torch.arange(-10, 10, 0.1)
    basis = GridBasis(H2, grid)
    density_ks, energy_ks = solve_scf(basis, H2.occ(), Lda1d())
    density_of, energy_of = solve_scf(basis, H2.occ(mode="OF"), Lda1d())
    assert_allclose(density_ks, density_of)
    assert_allclose(energy_ks, energy_of)


def test_h2_gauss():
    mol = gto.M(atom="H 0 0 0; H 0 0 1.1", basis="cc-pvdz", verbose=3)
    mf = dft.RKS(mol)
    mf.init_guess = "1e"
    mf.xc = "lda,pw"
    occ = torch.tensor([2])
    energy_true = mf.kernel()
    basis = GaussianBasis(mol)
    density, energy = solve_scf(basis, occ, LdaPw92())
    assert_allclose(energy, energy_true)


def test_h2_gauss_pbe():
    mol = gto.M(atom="H 0 0 0; H 0 0 1.1", basis="cc-pvdz", verbose=3)
    mf = dft.RKS(mol)
    mf.init_guess = "1e"
    mf.xc = "pbe"
    occ = torch.tensor([2])
    energy_true = mf.kernel()
    basis = GaussianBasis(mol)
    density, energy = solve_scf(basis, occ, PBE())
    assert_allclose(energy, energy_true)


def test_batched_scf():
    def get_chain(n, R):
        chain = torch.zeros(n)
        for i in range(n):
            chain[i] = i * R
        center = chain.mean()
        return chain - center

    grid = torch.arange(-10, 10, 0.1)
    R_list = [1.401118437, 2.5, 4]
    systems, gridbasis = [], []
    # Genetate some H2 and H3 systems:
    n = 2
    charges = torch.ones(n)
    for R in R_list:
        centers = get_chain(n, R)
        system = System(charges=charges, n_electrons=n, centers=centers)
        systems.append(system)
        gridbasis.append(GridBasis(system, grid))

    n = 3
    charges = torch.ones(n)
    for R in R_list:
        centers = get_chain(n, R)
        system = System(charges=charges, n_electrons=n, centers=centers)
        systems.append(system)
        gridbasis.append(GridBasis(system, grid))

    # Get batched system and grids.
    systembatch = SystemBatch(systems)
    batchgrid = BatchGridBasis(systembatch, grid)

    # Make two KS iterations:
    P_list, E_list = [], []
    for i, grid in enumerate(gridbasis):
        S, T, Vext = grid.get_core_integrals()
        S = GeneralizedDiagonalizer(S)
        F = T + Vext
        P, E = ks_iteration(F, S.X, systems[i].occ())

        V_H, V_xc, E_xc = grid.get_int_integrals(P, Lda1d())
        F = T + Vext + V_H + V_xc
        P, E = ks_iteration(F, S.X, systems[i].occ())

        P_list.append(P)
        E_list.append(E)

    # Get batched version of "ks_iteration".
    batch_ks_iteration = torch.vmap(ks_iteration, in_dims=(0, 0, 0))

    # Make two KS iterations with the batched version:
    Sb, T, Vext = batchgrid.get_core_integrals()
    X = torch.stack([GeneralizedDiagonalizer(S_i).X for S_i in Sb])
    F = T + Vext
    P, E = batch_ks_iteration(F, X, systembatch.occ())

    V_H, V_xc, E_xc = batchgrid.get_int_integrals(P, Lda1d())
    F = T + Vext + V_H + V_xc
    P, E = batch_ks_iteration(F, X, systembatch.occ())

    for i in range(systembatch.nbatch):
        assert_allclose(P_list[i], P[i])
        assert_allclose(E_list[i], E[i])


def test_batched__of_scf():
    def get_chain(n, R):
        chain = torch.zeros(n)
        for i in range(n):
            chain[i] = i * R
        center = chain.mean()
        return chain - center

    grid = torch.arange(-10, 10, 0.1)
    R_list = [1.401118437, 2.5, 4]
    systems, gridbasis = [], []
    # Genetate some H2 and H3 systems:
    n = 2
    charges = torch.ones(n)
    for R in R_list:
        centers = get_chain(n, R)
        system = System(charges=charges, n_electrons=n, centers=centers)
        systems.append(system)
        gridbasis.append(GridBasis(system, grid))

    n = 3
    charges = torch.ones(n)
    for R in R_list:
        centers = get_chain(n, R)
        system = System(charges=charges, n_electrons=n, centers=centers)
        systems.append(system)
        gridbasis.append(GridBasis(system, grid))

    # Get batched system and grids.
    systembatch = SystemBatch(systems)
    batchgrid = BatchGridBasis(systembatch, grid)

    # Make two KS iterations:
    P_list, E_list = [], []
    for i, grid in enumerate(gridbasis):
        S, T, Vext = grid.get_core_integrals()
        S = GeneralizedDiagonalizer(S)
        F = T + Vext
        P, E = ks_iteration(F, S.X, systems[i].occ("OF"))

        V_H, V_xc, E_xc = grid.get_int_integrals(P, Lda1d())
        F = T + Vext + V_H + V_xc
        P, E = ks_iteration(F, S.X, systems[i].occ("OF"))

        P_list.append(P)
        E_list.append(E)

    # Get batched version of "ks_iteration".
    batch_ks_iteration = torch.vmap(ks_iteration, in_dims=(0, 0, 0))

    # Make two KS iterations with the batched version:
    Sb, T, Vext = batchgrid.get_core_integrals()
    X = torch.stack([GeneralizedDiagonalizer(S_i).X for S_i in Sb])
    F = T + Vext
    P, E = batch_ks_iteration(F, X, systembatch.occ("OF"))

    V_H, V_xc, E_xc = batchgrid.get_int_integrals(P, Lda1d())
    F = T + Vext + V_H + V_xc
    P, E = batch_ks_iteration(F, X, systembatch.occ("OF"))

    for i in range(systembatch.nbatch):
        assert_allclose(P_list[i], P[i])
        assert_allclose(E_list[i], E[i])
