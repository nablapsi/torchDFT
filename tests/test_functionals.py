import torch
from torch.testing import assert_allclose

from torchdft.density import Density
from torchdft.functional import ComposedFunctional
from torchdft.gridbasis import GridBasis
from torchdft.kinetic_functionals import VonWeizsaecker
from torchdft.utils import System, gaussian
from torchdft.xc_functionals import Lda1d


class TestComposedFunctional:
    grid = torch.arange(-10, 10, 0.1)
    density = Density(gaussian(grid, 0, 1))
    # System and basis are declared in order to access basis._get_density_gradient
    # method.
    system = System(
        centers=torch.tensor([0]), charges=torch.tensor([1]), n_electrons=1, grid=grid
    )
    basis = GridBasis(system)
    batched_density = Density(torch.stack([gaussian(grid, 0, 1), gaussian(grid, 1, 1)]))

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
