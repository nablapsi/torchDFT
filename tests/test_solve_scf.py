from typing import List, Tuple

import torch
from pyscf import dft, gto
from torch import Tensor
from torch.testing import assert_allclose

from torchdft.errors import SCFNotConvergedError
from torchdft.gaussbasis import GaussianBasis
from torchdft.gridbasis import GridBasis
from torchdft.radialbasis import RadialBasis
from torchdft.scf import ks_iteration, solve_scf
from torchdft.utils import GeneralizedDiagonalizer, System, SystemBatch
from torchdft.xc_functionals import PBE, Lda1d, LdaPw92


def test_h2():
    charges = torch.tensor([1.0, 1.0], dtype=torch.float64)
    centers = torch.tensor([0.0, 1.401118437], dtype=torch.float64)
    nelectrons = 2
    grid = torch.arange(-10, 10, 0.1, dtype=torch.float64)
    H2 = System(nelectrons, charges, centers, grid)
    basis = GridBasis(H2)
    density, energy = solve_scf(basis, H2.occ(), Lda1d())
    assert_allclose(energy, -1.4046211)


def test_ks_of():
    charges = torch.tensor([1.0, 1.0])
    centers = torch.tensor([0.0, 1.401118437])
    nelectrons = 2
    grid = torch.arange(-10, 10, 0.1)
    H2 = System(nelectrons, charges, centers, grid)
    basis = GridBasis(H2)
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
    density, energy = solve_scf(basis, occ, LdaPw92(), mixer="pulay", use_xitorch=False)
    assert_allclose(energy, energy_true)


def test_h2_gauss_pbe():
    mol = gto.M(atom="H 0 0 0; H 0 0 1.1", basis="cc-pvdz", verbose=3)
    mf = dft.RKS(mol)
    mf.init_guess = "1e"
    mf.xc = "pbe"
    occ = torch.tensor([2])
    energy_true = mf.kernel()
    basis = GaussianBasis([mol])
    density, energy = solve_scf(basis, occ, PBE(), mixer="pulay")
    assert_allclose(energy[0], energy_true)


def test_batched_ks_iteration():
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
        system = System(charges=charges, n_electrons=n, centers=centers, grid=grid)
        systems.append(system)
        gridbasis.append(GridBasis(system))

    n = 3
    charges = torch.ones(n)
    for R in R_list:
        centers = get_chain(n, R)
        system = System(charges=charges, n_electrons=n, centers=centers, grid=grid)
        systems.append(system)
        gridbasis.append(GridBasis(system))

    # Get batched system and grids.
    systembatch = SystemBatch(systems)
    batchgrid = GridBasis(systembatch)

    for mode in ["KS", "OF"]:
        # Make two KS iterations:
        P_list, E_list = [], []
        for i, basis in enumerate(gridbasis):
            occ = systems[i].occ(mode)
            S, T, Vext = basis.get_core_integrals()
            S = GeneralizedDiagonalizer(S)
            F = T + Vext
            P, E = ks_iteration(F, S.X, occ)

            V_H, V_xc, E_xc = basis.get_int_integrals(P, Lda1d(), create_graph=False)
            F = T + Vext + V_H + V_xc
            P, E = ks_iteration(F, S.X, occ)

            P_list.append(P)
            E_list.append(E)

        # Make two KS iterations with the batched version:
        occ = systembatch.occ(mode)
        Sb, T, Vext = batchgrid.get_core_integrals()
        Sb = GeneralizedDiagonalizer(Sb)
        F = T + Vext
        P, E = ks_iteration(F, Sb.X, occ)

        V_H, V_xc, E_xc = batchgrid.get_int_integrals(P, Lda1d(), create_graph=False)
        F = T + Vext + V_H + V_xc
        P, E = ks_iteration(F, Sb.X, occ)

        for i in range(systembatch.nbatch):
            assert_allclose(P_list[i], P[i])
            assert_allclose(E_list[i], E[i])


def test_batched_solve_scf():
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
        system = System(charges=charges, n_electrons=n, centers=centers, grid=grid)
        systems.append(system)
        gridbasis.append(GridBasis(system))

    n = 3
    charges = torch.ones(n)
    for R in R_list:
        centers = get_chain(n, R)
        system = System(charges=charges, n_electrons=n, centers=centers, grid=grid)
        systems.append(system)
        gridbasis.append(GridBasis(system))

    # Get batched system and grids.
    systembatch = SystemBatch(systems)
    batchgrid = GridBasis(systembatch)

    for mode in ["KS", "OF"]:
        P_list, E_list = [], []
        for i, basis in enumerate(gridbasis):
            tape: List[Tuple[Tensor, Tensor]] = []
            occ = systems[i].occ(mode)
            try:
                solve_scf(basis, occ, Lda1d(), max_iterations=1, tape=tape)
            except SCFNotConvergedError:
                pass
            P_list.append(tape[-1][0])
            E_list.append(tape[-1][1])

        # Make two KS iterations with the batched version:
        occ = systembatch.occ(mode)
        tape = []
        try:
            solve_scf(batchgrid, occ, Lda1d(), max_iterations=1, tape=tape)
        except SCFNotConvergedError:
            pass
        for i in range(systembatch.nbatch):
            assert_allclose(P_list[i], tape[-1][0][i])
            assert_allclose(E_list[i], tape[-1][1][i])


def test_pulaydensity_ks_of():
    charges = torch.tensor([1.0, 1.0])
    centers = torch.tensor([0.0, 1.401118437])
    nelectrons = 2
    grid = torch.arange(-10, 10, 0.1)
    H2 = System(nelectrons, charges, centers, grid)
    basis = GridBasis(H2)
    density_ks, energy_ks = solve_scf(basis, H2.occ(), Lda1d(), mixer="pulaydensity")
    density_of, energy_of = solve_scf(
        basis, H2.occ(mode="OF"), Lda1d(), mixer="pulaydensity"
    )
    assert_allclose(density_ks, density_of)
    assert_allclose(energy_ks, energy_of)


def test_pulaydensity_h2_gauss():
    mol = gto.M(atom="H 0 0 0; H 0 0 1.1", basis="cc-pvdz", verbose=3)
    mf = dft.RKS(mol)
    mf.init_guess = "1e"
    mf.xc = "lda,pw"
    occ = torch.tensor([2])
    energy_true = mf.kernel()
    basis = GaussianBasis(mol)
    density, energy = solve_scf(
        basis, occ, LdaPw92(), mixer="pulaydensity", use_xitorch=False
    )
    assert_allclose(energy, energy_true)


def test_pulaydensity_h2_gauss_pbe():
    mol = gto.M(atom="H 0 0 0; H 0 0 1.1", basis="cc-pvdz", verbose=3)
    mf = dft.RKS(mol)
    mf.init_guess = "1e"
    mf.xc = "pbe"
    occ = torch.tensor([2])
    energy_true = mf.kernel()
    basis = GaussianBasis([mol])
    mixer_kwargs = {"precondition": False, "regularization": 0}
    density, energy = solve_scf(
        basis, occ, PBE(), mixer="pulaydensity", mixer_kwargs=mixer_kwargs
    )
    assert_allclose(energy[0], energy_true)


def test_Li_radialbasis():
    # True Li energy
    mol = gto.M(atom="Li 0 0 0", basis="cc-pv5z", verbose=3, spin=1)
    basis = GaussianBasis(mol)
    density, energy_truth = solve_scf(
        basis, torch.tensor([2, 1]), LdaPw92(), mixer="pulay", density_threshold=1e-9
    )
    # RadialBasis Li energy
    charges = torch.tensor(3.0, dtype=torch.float64)
    centers = torch.tensor(0.0, dtype=torch.float64)
    nelectrons = 3
    grid = torch.arange(1e-2, 10, 1e-2, dtype=torch.float64)
    Li = System(nelectrons, charges, centers, grid)
    basis = RadialBasis(Li)
    density, energy = solve_scf(
        basis, Li.occ(), LdaPw92(), mixer="pulay", density_threshold=1e-9
    )
    assert_allclose(energy, energy_truth, atol=1.6e-3, rtol=1e-4)
