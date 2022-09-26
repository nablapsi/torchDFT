import torch
from torch.testing import assert_allclose

from torchdft.density import Density
from torchdft.grid import Uniform1DGrid
from torchdft.gridbasis import (
    GridBasis,
    get_external_potential,
    get_external_potential_energy,
    get_functional_energy,
    get_functional_energy_potential,
    get_hartree_energy,
    get_hartree_potential,
    get_laplacian,
)
from torchdft.utils import System, gaussian, get_dx, soft_coulomb
from torchdft.xc_functionals import Lda1d


class TestFunctionals:
    def test_get_laplacian(self):
        grid_dim = 4
        lap = get_laplacian(grid_dim)

        true_lap = torch.tensor(
            [
                [-2.5, 4.0 / 3.0, -1.0 / 12.0, 0],
                [4.0 / 3.0, -2.5, 4.0 / 3.0, -1.0 / 12.0],
                [-1.0 / 12.0, 4.0 / 3.0, -2.5, 4.0 / 3.0],
                [0, -1.0 / 12.0, 4.0 / 3.0, -2.5],
            ]
        )
        assert_allclose(lap, true_lap)

    def test_get_hartree_energy(self):
        grid = torch.arange(-5, 5, 0.1)
        density = gaussian(grid, 1, 1)

        e1 = get_hartree_energy(density, grid, soft_coulomb)

        # Check against nested loop implementation:
        dx = get_dx(grid)
        e2 = torch.tensor(0.0)
        for n1, r1 in zip(density, grid):
            for n2, r2 in zip(density, grid):
                disp = r1 - r2
                e2 += n1 * n2 * soft_coulomb(disp)

        e2 *= 5e-1 * dx * dx
        assert_allclose(e1, e2)

    def test_get_hartree_potential(self):
        grid = torch.arange(-5, 5, 0.1)
        density = gaussian(grid, 1, 1)

        p1 = get_hartree_potential(density, grid, soft_coulomb)

        # Check against nested loop implementation:
        dx = get_dx(grid)
        p2 = torch.zeros(grid.size(0))
        for i, r in enumerate(grid):
            for n1, r1 in zip(density, grid):
                disp = r1 - r
                p2[i] += n1 * soft_coulomb(disp)

        p2 *= dx
        assert_allclose(p1, p2)

    def test_hartree_potential_ener(self):
        """
        The evaluated Hartree potential should be equal to the functional derivative
        of the Hartree energy with respect to the density.
        """
        grid = torch.arange(-5, 5, 0.1)
        dx = get_dx(grid)
        density = gaussian(grid, 1, 1)

        pot = get_hartree_potential(density, grid, soft_coulomb)

        if not density.requires_grad:
            density = density.requires_grad_()
        ener = get_hartree_energy(density, grid, soft_coulomb)
        ener.backward()
        assert_allclose(pot, density.grad / dx)

    def test_get_external_potential(self):
        grid = torch.arange(-5, 5, 0.1)

        charges = torch.tensor([-1, 1])
        centers = torch.tensor([0, 2])

        p1 = get_external_potential(charges, centers, grid, soft_coulomb)

        # Check against nested loop implementation:
        p2 = torch.zeros(grid.size(0))

        for i, r in enumerate(grid):
            for c1, r1 in zip(charges, centers):
                p2[i] -= c1 * soft_coulomb(r1 - r)

        assert_allclose(p1, p2)

    def test_external_potential_ener(self):
        """
        The evaluated external potential should be equal to the functional derivative
        of the external potential energy with respect to the density.
        """
        grid = torch.arange(-5, 5, 0.1)
        dx = get_dx(grid)
        density = gaussian(grid, 1, 1)

        charges = torch.tensor([-1, 1])
        centers = torch.tensor([0, 2])

        pot = get_external_potential(charges, centers, grid, soft_coulomb)

        if not density.requires_grad:
            density = density.requires_grad_()
        ener = get_external_potential_energy(pot, density, grid)
        ener.backward()
        assert_allclose(pot, density.grad / dx)

    def test_XC_potential_ener(self):
        """
        The evaluated XC potential should be equal to the functional derivative
        of the XC energy with respect to the density.
        """
        LDA = Lda1d()
        grid = torch.arange(-5, 5, 0.1)
        dx = get_dx(grid)
        density = Density(gaussian(grid, 1, 1))
        density.value = density.value.detach().requires_grad_()

        _, pot = get_functional_energy_potential(density, grid, LDA)

        density.value = density.value.detach().requires_grad_()
        ener = get_functional_energy(density, grid, LDA)
        ener.backward()
        assert_allclose(pot, density.value.grad / dx)

    def test_get_gradient(self):
        mean, std = 0, 1
        grid = Uniform1DGrid(dx=1e-1, end=20, reflection_symmetry=True)
        system = System(
            centers=torch.tensor([0]),
            Z=torch.tensor([1]),
        )
        basis = GridBasis(system, grid)
        density = gaussian(grid.grid, mean, std)
        den_der = basis._get_density_gradient(density)
        assert_allclose(
            den_der, -(grid.grid - mean) / std * density, atol=1e-5, rtol=1e-4
        )
