# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import torch

from .density import Density
from .kinetic_functionals import vW_energy
from .utils import exp_coulomb, get_dx
from .xc_functionals import Lda1d


class GridBasis:
    """Basis of equidistant 1D grid."""

    def __init__(
        self, system, grid, xc=Lda1d, kinetic=None, interaction_fn=exp_coulomb
    ):
        self.system = system
        self.grid = grid
        self.xc = xc()
        self.kinetic = kinetic
        self.interaction_fn = interaction_fn
        self.dx = get_dx(grid)
        self.E_nuc = (
            (
                (system.charges[:, None] * system.charges)
                * interaction_fn(system.centers[:, None] - system.centers)
            )
            .triu(diagonal=1)
            .sum()
        )

    def get_core_integrals(self):
        S = torch.full((len(self.grid),), self.dx, device=self.grid.device).diag_embed()
        T = self.dx * get_kinetic_matrix(self.grid)
        self.v_ext = get_external_potential(
            self.system.charges, self.system.centers, self.grid, self.interaction_fn
        )
        return S, T, self.dx * self.v_ext.diag_embed()

    def get_int_integrals(self, P):
        density = Density(P.diag())
        if self.kinetic or self.xc.requires_grad:
            density.grad = self._get_density_gradient(density.value)

        v_H = get_hartree_potential(density.value, self.grid, self.interaction_fn)
        E_xc, v_xc = get_XC_energy_potential(density, self.grid, self.xc)

        if self.kinetic:
            # For OF-DFT.
            T_s = get_kinetic_potential(density, self.grid, self.kinetic)
            E_K = get_kinetic_energy(
                density, self.grid, vW_energy
            ) + get_kinetic_energy(density, self.grid, self.kinetic)
            return (
                self.dx * T_s.diag_embed(),
                self.dx * v_H.diag_embed(),
                self.dx * v_xc.diag_embed(),
                E_K,
                E_xc,
            )
        else:
            # For KS-DFT.
            return (self.dx * v_H.diag_embed(), self.dx * v_xc.diag_embed(), E_xc)

    def _get_density_gradient(self, density):
        grad_operator = get_gradient(self.grid.size(0)) / self.dx
        return grad_operator.mv(density)


def get_gradient(grid_dim, device=None):
    """Finite difference approximation of gradient operator."""
    return (
        (2.0 / 3.0 * torch.ones(grid_dim - 1, device=device)).diag_embed(offset=1)
        + (-2.0 / 3.0 * torch.ones(grid_dim - 1, device=device)).diag_embed(offset=-1)
        + (-1.0 / 12.0 * torch.ones(grid_dim - 2, device=device)).diag_embed(offset=2)
        + (1.0 / 12.0 * torch.ones(grid_dim - 2, device=device)).diag_embed(offset=-2)
    )


def get_laplacian(grid_dim, device=None):
    """Finite difference approximation of Laplacian operator."""
    return (
        (-2.5 * torch.ones(grid_dim, device=device)).diag_embed()
        + (4.0 / 3.0 * torch.ones(grid_dim - 1, device=device)).diag_embed(offset=1)
        + (4.0 / 3.0 * torch.ones(grid_dim - 1, device=device)).diag_embed(offset=-1)
        + (-1.0 / 12.0 * torch.ones(grid_dim - 2, device=device)).diag_embed(offset=2)
        + (-1.0 / 12.0 * torch.ones(grid_dim - 2, device=device)).diag_embed(offset=-2)
    )


def get_kinetic_matrix(grid):
    """Kinetic operator matrix."""
    grid_dim = grid.size(0)
    dx = get_dx(grid)
    return -5e-1 * get_laplacian(grid_dim, device=grid.device) / (dx * dx)


def get_hartree_energy(density, grid, interaction_fn):
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

    grid_dim = grid.size(0)
    dx = get_dx(grid)

    n1 = torch.vstack((density,) * grid_dim)
    n2 = torch.swapdims(n1, 0, 1)
    r1 = torch.vstack((grid,) * grid_dim)
    r2 = torch.swapdims(r1, 0, 1)

    return 5e-1 * torch.sum(n1 * n2 * interaction_fn(r1 - r2)) * dx * dx


def get_hartree_potential(density, grid, interaction_fn):
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

    grid_dim = grid.size(0)
    dx = get_dx(grid)

    n1 = torch.vstack((density,) * grid_dim)
    r1 = torch.vstack((grid,) * grid_dim)
    r2 = torch.swapdims(r1, 0, 1)

    return torch.sum(n1 * interaction_fn(r1 - r2), axis=1) * dx


def get_external_potential_energy(external_potential, density, grid):
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
    return torch.dot(external_potential, density) * dx


def get_external_potential(charges, centers, grid, interaction_fn):
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

    ncharges = charges.size(0)
    grid_dim = grid.size(0)

    r1 = torch.vstack((grid,) * ncharges)
    r2 = torch.swapdims(torch.vstack((centers,) * grid_dim), 0, 1)
    c1 = torch.swapdims(torch.vstack((charges,) * grid_dim), 0, 1)

    return -torch.sum(c1 * interaction_fn(r1 - r2), axis=0)


def get_XC_energy(density, grid, xc):
    """Evaluate XC energy."""
    dx = get_dx(grid)

    return torch.dot(xc.functional(density), density.value) * dx


def get_XC_energy_potential(density, grid, xc):
    """Evaluate XC potential."""
    density = density.detach()
    density.value = density.value.requires_grad_()
    dx = get_dx(grid)
    E_xc = get_XC_energy(density, grid, xc)
    E_xc.backward()

    return E_xc.detach(), density.value.grad / dx


def get_kinetic_energy(density, grid, K_functional):
    """Evaluate the kinetic energy given a kinetic energy functional."""
    return K_functional(density, grid)


def get_kinetic_potential(density, grid, K_functional):
    """Evaluate functional derivative of kinetic energy wrt density."""
    density = density.detach()
    density.value = density.value.requires_grad_()
    dx = get_dx(grid)
    _ = get_kinetic_energy(density, grid, K_functional).backward()

    return density.value.grad / dx
