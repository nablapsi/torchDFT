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
        non_interacting: bool = False,
        reflection_symmetry: bool = False,
    ):
        super().__init__()
        self.non_interacting = non_interacting
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
        self.register_buffer("T", -5e-1 * self.get_laplacian())
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
        density = Density(self.density(P), self.grid, self.grid_weights)
        if functional.requires_grad:
            density.grad = self.get_density_gradient(P)
        if self.non_interacting:
            V_H = P.new_zeros(1)
        else:
            V_H = (
                get_hartree_potential(density.value, self.grid, self.interaction_fn)
            ).diag_embed()
        eps_func = functional(density)
        if functional.per_electron:
            eps_func = eps_func * density.value
        E_func = (eps_func * self.grid_weights).sum(-1)
        (v_func,) = torch.autograd.grad(
            eps_func.sum(), density.value, create_graph=create_graph
        )
        if torch.any(torch.isnan(v_func)):
            raise NanError()
        V_func = v_func.diag_embed()
        if not create_graph:
            E_func = E_func.detach()
            V_func = V_func.detach()
            V_H = V_H.detach()
        return V_H, V_func, E_func

    def get_density_gradient(self, P: Tensor) -> Tensor:
        density = self.density(P)
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
        metrics = {}
        Q, Q_ref = (self.quadrupole(x).detach() for x in [density, density_ref])
        mse = self.density_mse(density - density_ref)
        metrics["loss/quadrupole"] = ((Q - Q_ref) ** 2).mean(dim=0).sqrt()
        metrics["loss/density_rmse"] = (mse).mean(dim=0).sqrt().detach()
        return metrics

    def get_laplacian(self) -> Tensor:
        """Finite difference approximation of Laplacian operator."""
        grid_dim = self.grid.size(0)
        return (
            (-2.5 * self.grid.new_ones([grid_dim]).diag_embed())
            + (4.0 / 3.0 * self.grid.new_ones([grid_dim - 1]).diag_embed(offset=1))
            + (4.0 / 3.0 * self.grid.new_ones([grid_dim - 1]).diag_embed(offset=-1))
            + (-1.0 / 12.0 * self.grid.new_ones([grid_dim - 2]).diag_embed(offset=2))
            + (-1.0 / 12.0 * self.grid.new_ones([grid_dim - 2]).diag_embed(offset=-2))
        ) / self.grid_weights**2


def get_hartree_potential(
    density: Tensor,
    grid: Tensor,
    interaction_fn: Callable[[Tensor], Tensor] = exp_coulomb,
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
