import torch
from torch.testing import assert_allclose

from torchdft.density import Density
from torchdft.functional import ComposedFunctional
from torchdft.grid import Uniform1DGrid
from torchdft.gridbasis import GridBasis
from torchdft.kinetic_functionals import VonWeizsaecker
from torchdft.utils import System, gaussian
from torchdft.xc_functionals import Lda1d


class TestComposedFunctional:
    grid = Uniform1DGrid(end=10, dx=0.1, reflection_symmetry=True)
    density = Density(gaussian(grid.grid, 0, 1), grid.grid, grid.grid_weights)
    # System and basis are declared in order to access basis._get_density_gradient
    # method.
    system = System(centers=torch.tensor([0]), Z=torch.tensor([1]))
    basis = GridBasis(system, grid)
    batched_density = Density(
        torch.stack([gaussian(grid.grid, 0, 1), gaussian(grid.grid, 1, 1)]),
        grid.grid,
        grid.grid_weights,
    )

    def test_composed_functional(self):
        composed_functional = ComposedFunctional(
            functionals=[Lda1d(), VonWeizsaecker()]
        )
        assert composed_functional.requires_grad

        if composed_functional.requires_grad:
            self.density.grad = self.basis._get_density_gradient(self.density.value)

        epsilon_composed = composed_functional(self.density)
        epsilon1 = Lda1d()(self.density) * self.density.value
        epsilon2 = VonWeizsaecker()(self.density)
        assert_allclose(epsilon_composed, epsilon1 + epsilon2)

    def test_composed_functional_factors(self):
        factors = [0.5, 0.2]

        composed_functional = ComposedFunctional(
            functionals=[Lda1d(), VonWeizsaecker()], factors=factors
        )
        assert composed_functional.requires_grad

        if composed_functional.requires_grad:
            self.density.grad = self.basis._get_density_gradient(self.density.value)

        epsilon_composed = composed_functional(self.density)
        epsilon1 = Lda1d()(self.density) * self.density.value
        epsilon2 = VonWeizsaecker()(self.density)
        assert_allclose(epsilon_composed, factors[0] * epsilon1 + factors[1] * epsilon2)

    def test_composed_functional_batched(self):
        factors = [0.5, 0.2]

        composed_functional = ComposedFunctional(
            functionals=[Lda1d(), VonWeizsaecker()], factors=factors
        )
        assert composed_functional.requires_grad

        if composed_functional.requires_grad:
            self.batched_density.grad = self.basis._get_density_gradient(
                self.batched_density.value
            )

        epsilon_composed = composed_functional(self.batched_density)
        epsilon1 = Lda1d()(self.batched_density) * self.batched_density.value
        epsilon2 = VonWeizsaecker()(self.batched_density)
        assert_allclose(epsilon_composed, factors[0] * epsilon1 + factors[1] * epsilon2)
