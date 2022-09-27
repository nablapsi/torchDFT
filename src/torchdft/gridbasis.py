# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Callable, Dict, Tuple, Union

import torch
from torch import Tensor

from .basis import Basis
from .density import Density
from .errors import NanError
from .functional import Functional
from .grid import Grid
from .utils import System, SystemBatch, exp_coulomb, fin_diff_matrix, get_dx


class GridBasis(Basis):
    """Basis of equidistant 1D grid."""

    centers: Tensor
    Z: Tensor

    def __init__(
        self,
        system: Union[System, SystemBatch],
        grid: Grid,
        interaction_fn: Callable[[Tensor], Tensor] = exp_coulomb,
        reflection_symmetry: bool = False,
    ):
        super().__init__()
        self.reflection_symmetry = reflection_symmetry
        self.system = system
        self.interaction_fn = interaction_fn
        self.register_buffer("grid", grid.grid)
        self.register_buffer("grid_weights", grid.grid_weights)
        self.register_buffer("dv", grid.dv)
        self.register_buffer("centers", self.system.centers)
        self.register_buffer("Z", self.system.Z)
        self.register_buffer(
            "E_nuc",
            (
                (system.Z[..., None, :] * system.Z[..., None])
                * interaction_fn(
                    system.centers[..., None, :] - system.centers[..., None]
                )
            )
            .triu(diagonal=1)
            .sum((-2, -1)),
        )
        self.register_buffer("T", get_kinetic_matrix(self.grid))
        self.register_buffer(
            "V_ext",
            (
                -(
                    self.system.Z[..., None]
                    * self.interaction_fn(self.grid - self.centers[..., None])
                ).sum(-2)
            ).diag_embed(),
        )
        self.register_buffer("S", self.grid.new_ones(self.grid.size()).diag_embed())
        self.register_buffer(
            "grad_operator",
            fin_diff_matrix(self.grid.shape[0], 5, 1, dtype=self.grid.dtype)
            / self.grid_weights,
        )

    def get_core_integrals(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.S, self.T, self.V_ext

    def get_int_integrals(
        self,
        P: Tensor,
        functional: Functional,
        create_graph: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Evaluate Hartree and extra potential contributions.

        Args:
            P: Float torch array of dimension (grid_dim, grid_dim)
              holding the density matrix.
            functional: Functional or ComposedFunctional.
              XC, kinetic, or linear combination of functionals.
            create_graph: Bool.
        """
        if not P.requires_grad:
            P = P.detach().requires_grad_()
        density = Density(self.density(P))
        if functional.requires_grad:
            density.grad = self._get_density_gradient(density.value)

        V_H = get_hartree_potential(density.value, self.grid, self.interaction_fn).diag_embed()
        E_func, v_func = get_functional_energy_potential(
            density, self.grid, functional, create_graph
        )
        return (
            V_H,
            v_func.diag_embed(),
            E_func,
        )

    def _get_density_gradient(self, density: Tensor) -> Tensor:
        return torch.einsum("ij, ...j -> ...i", self.grad_operator, density)

    def density_mse(self, density: Tensor) -> Tensor:
        return density.pow(2).sum(dim=-1) * self.grid_weights

    def density(self, P: Tensor) -> Tensor:
        if self.reflection_symmetry:
            P = (P + P.flip(-1, -2)) / 2
        return P.diagonal(dim1=-2, dim2=-1) / self.grid_weights

    def quadrupole(self, density: Tensor) -> Tensor:
        Q_el = -(self.grid**2 * density).sum(-1) * self.grid_weights
        Q_nuc = (self.centers**2 * self.Z).sum(-1)
        return Q_el + Q_nuc

    def density_metrics_fn(
        self, density: Tensor, density_ref: Tensor
    ) -> Dict[str, Tensor]:
        Q, Q_ref = (self.quadrupole(x).detach() for x in [density, density_ref])
        return {"loss/quadrupole": ((Q - Q_ref) ** 2).mean().sqrt()}


def get_laplacian(grid_dim: int, device: torch.device = None) -> Tensor:
    """Finite difference approximation of Laplacian operator."""
    return (
        (-2.5 * torch.ones(grid_dim, device=device)).diag_embed()
        + (4.0 / 3.0 * torch.ones(grid_dim - 1, device=device)).diag_embed(offset=1)
        + (4.0 / 3.0 * torch.ones(grid_dim - 1, device=device)).diag_embed(offset=-1)
        + (-1.0 / 12.0 * torch.ones(grid_dim - 2, device=device)).diag_embed(offset=2)
        + (-1.0 / 12.0 * torch.ones(grid_dim - 2, device=device)).diag_embed(offset=-2)
    )


def get_kinetic_matrix(grid: Tensor) -> Tensor:
    """Kinetic operator matrix."""
    grid_dim = grid.size(0)
    dx = get_dx(grid)
    return -5e-1 * get_laplacian(grid_dim, device=grid.device) / (dx * dx)


def get_hartree_energy(
    density: Tensor, grid: Tensor, interaction_fn: Callable[[Tensor], Tensor]
) -> Tensor:
    r"""Evaluate Hartree energy.

    Get Hartree energy evaluated as:
    0.5 \int \int n(r) n(r') interaction_function(r, r') dr dr'

    Args:
        density: Float torch array of dimension (grid_dim,) holding the density
          at each spatial point.
        grid: Float torch array of dimension (grid_dim,).
        interaction_fn: Function that, provided the displacements returns a float
          torch array.

    Returns:
        Float. Hartree energy.
    """

    dx = get_dx(grid)

    return (
        5e-1
        * (
            density[..., None, :]
            * density[..., None]
            * interaction_fn(grid[:, None] - grid)
        ).sum((-2, -1))
        * dx
        * dx
    )


def get_hartree_potential(
    density: Tensor, grid: Tensor, interaction_fn: Callable[[Tensor], Tensor]
) -> Tensor:
    r"""Evaluate Hartree potential.

    Get Hartree potential evaluated as:
    \int n(r') interaction_function(r, r') dr'

    Args:
        density: Float torch array of dimension (grid_dim,) holding the density
          at each spatial point.
        grid: Float torch array of dimension (grid_dim,).
        interaction_fn: Function that, provided the displacements returns a float
          torch array.

    Returns:
        Float torch array of dimension (grid_dim,) holding the hartree potential
          energy at each spatial point.
    """

    dx = get_dx(grid)

    return (density[..., None, :] * interaction_fn(grid[:, None] - grid)).sum(-1) * dx


def get_functional_energy(
    density: Density, grid: Tensor, functional: Functional
) -> Tensor:
    """Evaluate functional energy."""
    dx = get_dx(grid)
    eps = functional(density)
    if functional.per_electron:
        eps = eps * density.value
    return eps.sum(-1) * dx


def get_functional_energy_potential(
    density: Density,
    grid: Tensor,
    functional: Functional,
    create_graph: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Evaluate functional potential."""
    density.value = torch.where(
        density.value <= 0.0,
        density.value.new_tensor(1e-100),
        density.value,
    )
    dx = get_dx(grid)
    E_func = get_functional_energy(density, grid, functional)
    (v_func,) = torch.autograd.grad(
        E_func.sum() / dx, density.value, create_graph=create_graph
    )
    if not create_graph:
        E_func = E_func.detach()
    if torch.any(torch.isnan(v_func)):
        raise NanError()
    return E_func, v_func
