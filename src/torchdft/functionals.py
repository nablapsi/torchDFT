# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import torch

from .utils import get_dx


def get_laplacian(grid_dim):
    """Finite difference approximation of Laplacian operator."""
    return (
        torch.diag_embed(-2.5 * torch.ones((grid_dim)))
        + torch.diag_embed(4.0 / 3.0 * torch.ones(grid_dim - 1), offset=1)
        + torch.diag_embed(4.0 / 3.0 * torch.ones(grid_dim - 1), offset=-1)
        + torch.diag_embed(-1.0 / 12.0 * torch.ones(grid_dim - 2), offset=2)
        + torch.diag_embed(-1.0 / 12.0 * torch.ones(grid_dim - 2), offset=-2)
    )


def get_kinetic_matrix(grid):
    """Kinetic operator matrix."""
    grid_dim = grid.size(0)
    dx = get_dx(grid)
    return -5e-1 * get_laplacian(grid_dim) / (dx * dx)


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


def get_XC_energy(density, grid, XC_energy_density):
    """Evaluate XC energy."""
    dx = get_dx(grid)

    return torch.dot(XC_energy_density(density), density) * dx


def get_XC_potential(density, grid, XC_energy_density):
    """Evaluate XC potential."""
    # This copy is needed since it will load gradients on density Tensor otherwise.
    cdensity = density.clone().detach()
    cdensity.requires_grad = True
    cdensity.grad = None
    dx = get_dx(grid)
    _ = get_XC_energy(cdensity, grid, XC_energy_density).backward()

    return cdensity.grad / dx

    density.requires_grad = True
    dx = get_dx(grid)
    _ = get_XC_energy(density, grid, XC_energy_density).backward()

    return density.grad / dx
