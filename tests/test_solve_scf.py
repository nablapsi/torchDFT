import torch
from pyscf import dft, gto
from torch.testing import assert_allclose

from torchdft.errors import SCFNotConvergedError
from torchdft.gaussbasis import GaussianBasis
from torchdft.grid import RadialGrid, Uniform1DGrid
from torchdft.gridbasis import GridBasis
from torchdft.radialbasis import RadialBasis
from torchdft.scf import RKS
from torchdft.utils import System, SystemBatch
from torchdft.xc_functionals import PBE, Lda1d, LdaPw92

torch.set_default_dtype(torch.double)


def test_h2():
    Z = torch.tensor([1, 1])
    centers = torch.tensor([-0.7005592185, 0.7005592185], dtype=torch.float64)
    grid = Uniform1DGrid(start=-10, end=10, dx=0.1)
    H2 = System(Z=Z, centers=centers)
    basis = GridBasis(H2, grid, reflection_symmetry=True)
    solver = RKS(basis, H2.occ("KS"), Lda1d())
    sol = solver.solve()
    assert_allclose(sol.E[0], -1.4045913)


def test_ks_of():
    Z = torch.tensor([1.0, 1.0])
    centers = torch.tensor([0.0, 1.401118437])
    grid = Uniform1DGrid(end=10, dx=0.1, reflection_symmetry=True)
    H2 = System(Z=Z, centers=centers)
    basis = GridBasis(H2, grid, reflection_symmetry=True)
    solver = RKS(basis, H2.occ("KS"), Lda1d())
    sol_ks = solver.solve()
    solver = RKS(basis, H2.occ("OF"), Lda1d())
    sol_of = solver.solve()
    assert_allclose(sol_ks.P, sol_of.P)
    assert_allclose(sol_ks.E, sol_of.E)


def test_h2_gauss():
    mol = gto.M(atom="H 0 0 0; H 0 0 1.1", basis="cc-pvdz", verbose=3)
    mf = dft.RKS(mol)
    mf.init_guess = "1e"
    mf.xc = "lda,pw"
    occ = torch.tensor([[2]])
    energy_true = mf.kernel()
    basis = GaussianBasis(mol)
    solver = RKS(basis, occ, LdaPw92())
    sol = solver.solve()
    assert_allclose(sol.E[0], energy_true)


def test_h2_gauss_pbe():
    mol = gto.M(atom="H 0 0 0; H 0 0 1.1", basis="cc-pvdz", verbose=3)
    mf = dft.RKS(mol)
    mf.init_guess = "1e"
    mf.xc = "pbe"
    occ = torch.tensor([[2]])
    energy_true = mf.kernel()
    basis = GaussianBasis(mol)
    solver = RKS(basis, occ, PBE())
    sol = solver.solve(mixer="pulay", density_threshold=1e-9)
    assert_allclose(sol.E[0], energy_true)


def test_batched_ks_iteration():
    def get_chain(n, R):
        chain = torch.zeros(n)
        for i in range(n):
            chain[i] = i * R
        center = chain.mean()
        return chain - center

    grid = Uniform1DGrid(end=10, dx=0.1, reflection_symmetry=True)
    R_list = [1.401118437, 2.5, 4]
    systems, gridbasis = [], []
    # Genetate some H2 and H3 systems:
    n = 2
    charges = torch.ones(n)
    for R in R_list:
        centers = get_chain(n, R)
        system = System(Z=charges, centers=centers)
        systems.append(system)
        gridbasis.append(GridBasis(system, grid))

    n = 3
    charges = torch.ones(n)
    for R in R_list:
        centers = get_chain(n, R)
        system = System(Z=charges, centers=centers)
        systems.append(system)
        gridbasis.append(GridBasis(system, grid))

    # Get batched system and grids.
    systembatch = SystemBatch(systems)
    batchgrid = GridBasis(systembatch, grid)

    for mode in ["KS", "OF"]:
        # Make two KS iterations:
        P_list, E_list = [], []
        for i, basis in enumerate(gridbasis):
            occ = systems[i].occ(mode)
            solver = RKS(basis, occ, Lda1d())
            try:
                sol = solver.solve(mixer="linear", max_iterations=2)
                P_list.append(sol.P)
                E_list.append(sol.E)
            except SCFNotConvergedError as e:
                P_list.append(e.sol.P)
                E_list.append(e.sol.E)

        # Make two KS iterations with the batched version:
        occ = systembatch.occ(mode)
        solver = RKS(batchgrid, occ, Lda1d())
        try:
            sol = solver.solve(mixer="linear", max_iterations=2)
            P = sol.P
            E = sol.E
        except SCFNotConvergedError as e:
            P = e.sol.P
            E = e.sol.E
        for i in range(systembatch.nbatch):
            assert_allclose(P_list[i][0], P[i])
            assert_allclose(E_list[i][0], E[i])


def test_pulaydensity_ks_of():
    Z = torch.tensor([1.0, 1.0])
    centers = torch.tensor([0.0, 1.401118437])
    grid = Uniform1DGrid(end=10, dx=0.1, reflection_symmetry=True)
    H2 = System(Z=Z, centers=centers)
    basis = GridBasis(H2, grid)
    solver = RKS(basis, H2.occ("KS"), Lda1d())
    sol_ks = solver.solve(mixer="pulaydensity")
    solver = RKS(basis, H2.occ("OF"), Lda1d())
    sol_of = solver.solve(mixer="pulaydensity")
    assert_allclose(sol_ks.P, sol_of.P)
    assert_allclose(sol_ks.E, sol_of.E)


def test_pulaydensity_ks_of_noxitorch():
    Z = torch.tensor([1.0, 1.0])
    centers = torch.tensor([0.0, 1.401118437])
    grid = Uniform1DGrid(end=10, dx=0.1, reflection_symmetry=True)
    H2 = System(Z=Z, centers=centers)
    basis = GridBasis(H2, grid)
    solver = RKS(basis, H2.occ("KS"), Lda1d())
    sol_ks = solver.solve(use_xitorch=False, mixer="pulaydensity")
    solver = RKS(basis, H2.occ("OF"), Lda1d())
    sol_of = solver.solve(use_xitorch=False, mixer="pulaydensity")
    assert_allclose(sol_ks.P, sol_of.P)
    assert_allclose(sol_ks.E, sol_of.E)


def test_pulaydensity_h2_gauss():
    mol = gto.M(atom="H 0 0 0; H 0 0 1.1", basis="cc-pvdz", verbose=3)
    mf = dft.RKS(mol)
    mf.init_guess = "1e"
    mf.xc = "lda,pw"
    occ = torch.tensor([[2]])
    energy_true = mf.kernel()
    basis = GaussianBasis(mol)
    solver = RKS(basis, occ, LdaPw92())
    sol = solver.solve(mixer="pulaydensity", use_xitorch=True)
    assert_allclose(sol.E[0], energy_true)


def test_pulaydensity_h2_gauss_noxitorch():
    mol = gto.M(atom="H 0 0 0; H 0 0 1.1", basis="cc-pvdz", verbose=3)
    mf = dft.RKS(mol)
    mf.init_guess = "1e"
    mf.xc = "lda,pw"
    occ = torch.tensor([2])
    energy_true = mf.kernel()
    basis = GaussianBasis(mol)
    solver = RKS(basis, occ, LdaPw92())
    sol = solver.solve(mixer="pulaydensity", use_xitorch=False)
    assert_allclose(sol.E[0], energy_true)


def test_pulaydensity_h2_gauss_pbe():
    mol = gto.M(atom="H 0 0 0; H 0 0 1.1", basis="cc-pvdz", verbose=3)
    mf = dft.RKS(mol)
    mf.init_guess = "1e"
    mf.xc = "pbe"
    occ = torch.tensor([[2]])
    energy_true = mf.kernel()
    basis = GaussianBasis(mol)
    mixer_kwargs = {"precondition": False, "regularization": 0}
    solver = RKS(
        basis,
        occ,
        PBE(),
    )
    sol = solver.solve(
        mixer="pulaydensity",
        mixer_kwargs=mixer_kwargs,
        use_xitorch=True,
    )
    assert_allclose(sol.E[0], energy_true)


def test_pulaydensity_h2_gauss_pbe_noxitorch():
    mol = gto.M(atom="H 0 0 0; H 0 0 1.1", basis="cc-pvdz", verbose=3)
    mf = dft.RKS(mol)
    mf.init_guess = "1e"
    mf.xc = "pbe"
    occ = torch.tensor([2])
    energy_true = mf.kernel()
    basis = GaussianBasis(mol)
    mixer_kwargs = {"precondition": False, "regularization": 0}
    solver = RKS(
        basis,
        occ,
        PBE(),
    )
    sol = solver.solve(
        mixer="pulaydensity",
        mixer_kwargs=mixer_kwargs,
        use_xitorch=False,
    )
    assert_allclose(sol.E[0], energy_true)


def test_Li_radialbasis():
    # True Li energy
    mol = gto.M(atom="Li 0 0 0", basis="cc-pv5z", verbose=3, spin=1)
    basis = GaussianBasis(mol)
    solver = RKS(basis, torch.tensor([[2, 1]]), LdaPw92())
    sol_truth = solver.solve(mixer="pulay", density_threshold=1e-9)

    # RadialBasis Li energy
    grid = RadialGrid(end=10, dx=1e-2)
    Li = System(Z=torch.tensor([3]), centers=torch.tensor([0]))
    basis = RadialBasis(Li, grid)
    solver = RKS(
        basis,
        Li.occ("aufbau"),
        LdaPw92(),
    )
    sol = solver.solve(
        mixer="pulay",
        density_threshold=1e-9,
        extra_fock_channel=True,
    )
    assert_allclose(sol.E, sol_truth.E, atol=1.6e-3, rtol=1e-4)


def test_batched_radialbasis():
    grid = RadialGrid(end=10.0, dx=1e0)
    # Li
    Li = System(Z=torch.tensor([3]), centers=torch.tensor([0]))
    # C
    C = System(Z=torch.tensor([6]), centers=torch.tensor([0]))
    batch = SystemBatch([Li, C])
    Libasis = RadialBasis(Li, grid)
    Cbasis = RadialBasis(C, grid)
    batchedbasis = RadialBasis(batch, grid)
    solver = RKS(
        Libasis,
        Li.occ("aufbau"),
        functional=LdaPw92(),
    )
    sol_Li = solver.solve(
        mixer="pulaydensity",
        density_threshold=1e-9,
        extra_fock_channel=True,
    )
    solver = RKS(
        Cbasis,
        C.occ("aufbau"),
        functional=LdaPw92(),
    )
    sol_C = solver.solve(
        mixer="pulaydensity",
        density_threshold=1e-9,
        extra_fock_channel=True,
    )
    solver = RKS(
        batchedbasis,
        batch.occ("aufbau"),
        functional=LdaPw92(),
    )
    sol = solver.solve(
        mixer="pulaydensity",
        density_threshold=1e-9,
        extra_fock_channel=True,
    )

    assert_allclose(sol_Li.E[0], sol.E[0])
    assert_allclose(sol_Li.P[0], sol.P[0])
    assert_allclose(sol_C.E[0], sol.E[1])
    assert_allclose(sol_C.P[0], sol.P[1])
