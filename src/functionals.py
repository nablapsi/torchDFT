import torch
from utils import get_dx

def get_hartree_energy(density, grid, interaction_fn):
    """
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

    n1 = torch.vstack((density,)*grid_dim)
    n2 = torch.swapdims(n1, 0, 1)
    r1 = torch.vstack((grid,)*grid_dim)
    r2 = torch.swapdims(r1, 0, 1)

    return 5e-1 * torch.sum(n1 * n2 * interaction_fn(r1 - r2)) * dx * dx

def get_hartree_potential(density, grid, interaction_fn):
    """
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

    n1 = torch.vstack((density,)*grid_dim)
    r1 = torch.vstack((grid,)*grid_dim)
    r2 = torch.swapdims(r1, 0, 1)

    return 5e-1 * torch.sum(n1 * interaction_fn(r1 - r2), axis=1) * dx
