# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import math

import torch


def get_dx(grid):
    """
    Given a grid as a 1D array returns the spacing between grid points.
    Args:
        grid: Float torch array of dimension (grid_dim,).

    Returns:
        Float.
    """
    grid_dim = grid.size(0)
    return (torch.amax(grid) - torch.amin(grid)) / grid_dim


def gaussian(x, center, sigma):
    return (
        1e0
        / torch.sqrt(torch.tensor(2 * math.pi))
        * torch.exp(-5e-1 * ((x - center) / sigma) ** 2)
        / sigma
    )


def soft_coulomb(r):
    return 1 / torch.sqrt(r ** 2 + 1e0)
