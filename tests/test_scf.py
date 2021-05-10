import torch
from torch.testing import assert_allclose

from torchdft.functionals import (
    get_hartree_potential,
    get_kinetic_matrix,
    get_XC_potential,
)
from torchdft.scf import (
    get_density_from_wf,
    get_effective_potential,
    get_hamiltonian_matrix,
    get_total_eigen_ener,
)
from torchdft.utils import exp_coulomb, gaussian
from torchdft.xc_functionals import exponential_coulomb_LDA_XC_energy_density


class TestSCF:
    def test_get_effective_potential(self):
        grid = torch.arange(-5, 5, 1)

        density = gaussian(grid, 0.0, 1.0)

        vext = torch.Tensor(
            [
                -0.1317,
                -0.2003,
                -0.3046,
                -0.4632,
                -0.7044,
                -1.0713,
                -0.7044,
                -0.4632,
                -0.3046,
                -0.2003,
            ]
        )
        vH = get_hartree_potential(density, grid, exp_coulomb)
        vXC = get_XC_potential(density, grid, exponential_coulomb_LDA_XC_energy_density)
        v = vext + vH + vXC

        assert_allclose(
            get_effective_potential(
                vext,
                density,
                grid,
                exp_coulomb,
                exponential_coulomb_LDA_XC_energy_density,
            ),
            v,
        )

    def test_get_hamiltonian_matrix(self):
        grid = torch.arange(-5, 5, 1)
        vext = torch.Tensor(
            [
                -0.1317,
                -0.2003,
                -0.3046,
                -0.4632,
                -0.7044,
                -1.0713,
                -0.7044,
                -0.4632,
                -0.3046,
                -0.2003,
            ]
        )

        H = get_hamiltonian_matrix(grid, vext)
        Htrue = get_kinetic_matrix(grid)
        for i in range(Htrue.size(0)):
            Htrue[i, i] += vext[i]

        assert_allclose(H, Htrue)

    def test_get_density_from_wf(self):
        nelectrons = 4
        grid = torch.linspace(-2, 2, 5)
        wf = torch.Tensor([[0.0, 0.0, 1.0, 0.0, 0.0], [-1, 0.0, 0.0, 1.0, 0.0]])

        assert_allclose(
            get_density_from_wf(nelectrons, grid, torch.swapdims(wf, 0, 1)),
            torch.Tensor([1.0, 0.0, 2.0, 1.0, 0.0]),
        )

    def test_get_total_eigen_ener(self):
        nelectrons = 4
        eigener = torch.Tensor([1.0, 2.0, 3.0, 4.0])

        assert_allclose(get_total_eigen_ener(nelectrons, eigener), torch.tensor(6.0))
