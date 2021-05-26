# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Callable, Tuple, Union

import torch
from torch import Tensor

from .basis import Basis
from .density import Density
from .functional import Functional
from .utils import System, SystemBatch, exp_coulomb, get_dx


class GridBasis(Basis):
    """Basis of equidistant 1D grid."""

    def __init__(
        self,
        system: Union[System, SystemBatch],
        grid: Tensor,
        interaction_fn: Callable[[Tensor], Tensor] = exp_coulomb,
    ):
        self.system = system
        self.grid = grid
        self.interaction_fn = interaction_fn
        self.dx = get_dx(grid)

        self.E_nuc = (
            (
                (system.charges[..., None, :] * system.charges[..., None])
                * interaction_fn(
                    system.centers[..., None, :] - system.centers[..., None]
                )
            )
            .triu(diagonal=1)
            .sum()
        )

    def get_core_integrals(self) -> Tuple[Tensor, Tensor, Tensor]:
        T = self.dx * get_kinetic_matrix(self.grid)
        self.v_ext = get_external_potential(
            self.system.charges, self.system.centers, self.grid, self.interaction_fn
        )
        S = torch.full_like(self.v_ext, self.dx, device=self.grid.device).diag_embed()
        return S, T, self.dx * self.v_ext.diag_embed()

    def get_int_integrals(
        self, P: Tensor, xc_functional: Functional
    ) -> Tuple[Tensor, Tensor, Tensor]:
        density = Density(torch.diagonal(P, dim1=-2, dim2=-1))
        if xc_functional.requires_grad:
            density.grad = self._get_density_gradient(density.value)

        v_H = get_hartree_potential(density.value, self.grid, self.interaction_fn)
        E_xc, v_xc = get_XC_energy_potential(density, self.grid, xc_functional)
        return self.dx * v_H.diag_embed(), self.dx * v_xc.diag_embed(), E_xc

    def _get_density_gradient(self, density: Tensor) -> Tensor:
        grad_operator = get_gradient(self.grid.size(0)) / self.dx
        return grad_operator.mv(density)

    def density_rms(self, P: Tensor) -> Tensor:
        dx = get_dx(self.grid)
        density = torch.diagonal(P, dim1=-2, dim2=-1)
        return density.pow(2).sum(dim=-1) * dx


def get_gradient(grid_dim: int, device: torch.device = None) -> Tensor:
    """Finite difference approximation of gradient operator."""
    return (
        (2.0 / 3.0 * torch.ones(grid_dim - 1, device=device)).diag_embed(offset=1)
        + (-2.0 / 3.0 * torch.ones(grid_dim - 1, device=device)).diag_embed(offset=-1)
        + (-1.0 / 12.0 * torch.ones(grid_dim - 2, device=device)).diag_embed(offset=2)
        + (1.0 / 12.0 * torch.ones(grid_dim - 2, device=device)).diag_embed(offset=-2)
    )


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
    0.5 \int n(r') interaction_function(r, r') dr'

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

    return (
        density[..., None, :] * interaction_fn(grid[..., None, :] - grid[..., None])
    ).sum(-1) * dx


def get_external_potential_energy(
    external_potential: Tensor, density: Tensor, grid: Tensor
) -> Tensor:
    r"""Evaluate external potential energy.

    Get external potential energy evaluated as:
    \int v_ext(r) n(r) dr

    Args:
        external_potential: Float torch array of dimension (grid_dim,)
          holding the external potential at each grid point.
        density: Float torch array of dimension (grid_dim,) holding the density
          at each spatial point.
        grid: Float torch array of dimension (grid_dim,).

    Returns:
        Float. External potential energy.
    """

    dx = get_dx(grid)
    return (external_potential * density).sum(-1) * dx


def get_external_potential(
    charges: Tensor,
    centers: Tensor,
    grid: Tensor,
    interaction_fn: Callable[[Tensor], Tensor],
) -> Tensor:
    r"""Evaluate external potential.

    Get external potential evaluated as:
    \sum_{n=1}^N - Z_n \cdot interaction_function(r, r')

    Args:
        charges: Float torch array of dimension (ncharges,) holding the charges
          of each nucleus.
        centers: Float torch array of dimension (ncharges,) holding the positions
          of each nucleus.
        grid: Float torch array of dimension (grid_dim,).
        interaction_fn: Function that, provided the displacements returns a float
          torch array.

    Returns:
        Float torch array of dimension (grid_dim,) holding the external potential
          energy at each spatial point.
    """

    return -(
        charges[..., None] * interaction_fn(grid[..., None, :] - centers[..., None])
    ).sum(-2)


def get_XC_energy(density: Density, grid: Tensor, xc: Functional) -> Tensor:
    """Evaluate XC energy."""
    dx = get_dx(grid)

    return (xc(density) * density.value).sum(-1) * dx


def get_XC_energy_potential(
    density: Density, grid: Tensor, xc: Functional
) -> Tuple[Tensor, Tensor]:
    """Evaluate XC potential."""
    density = density.detach()
    density.value = density.value.requires_grad_()
    dx = get_dx(grid)
    E_xc = get_XC_energy(density, grid, xc)
    (E_xc).sum().backward()

    return E_xc.detach(), density.value.grad / dx
