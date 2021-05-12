# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import torch

from .utils import get_dx


def TF_energy_1d(density, grid, A=0.3):
    """Evaluate the Thomas Fermi kinetic energy in one dimension.

    https://journals.aps.org/pra/pdf/10.1103/PhysRevA.60.2285
    """
    dx = get_dx(grid)
    return A * (density ** 2).sum() * dx


def vW_energy(density, grid):
    """Evaluate the von Weizsaecker kinetic energy."""

    # NOTE: If get_gradient was defined in gridbasis.py there is a circular
    # import between this two files.
    def get_gradient(grid_dim, device=None):
        """Finite difference approximation of gradient operator."""
        return (
            (2.0 / 3.0 * torch.ones(grid_dim - 1, device=device)).diag_embed(offset=1)
            + (-2.0 / 3.0 * torch.ones(grid_dim - 1, device=device)).diag_embed(
                offset=-1
            )
            + (-1.0 / 12.0 * torch.ones(grid_dim - 2, device=device)).diag_embed(
                offset=2
            )
            + (1.0 / 12.0 * torch.ones(grid_dim - 2, device=device)).diag_embed(
                offset=-2
            )
        )

    dx = get_dx(grid)
    grid_dim = grid.size(0)
    grad_operator = get_gradient(grid_dim, device=grid.device) / dx
    return 1.0 / 8.0 * (grad_operator.mv(density) ** 2 / density).sum() * dx
