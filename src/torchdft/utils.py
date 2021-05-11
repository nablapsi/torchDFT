# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import math
from collections import namedtuple

import torch

from torchdft import constants

# TODO: Find better way to express the system.
System = namedtuple(
    "System", ["nelectrons", "charges", "centers"]
)


def get_dx(grid):
    """Get grid spacing.

    Given a grid as a 1D array returns the spacing between grid points.
    Args:
        grid: Float torch array of dimension (grid_dim,).

    Returns:
        Float.
    """
    grid_dim = grid.size(0)
    return (torch.amax(grid) - torch.amin(grid)) / (grid_dim - 1)


def gaussian(x, mean, sigma):
    """Gaussian function.

    Evaluates a gaussian function with mean = mean and std = sigma over
    an array of positions x
    """
    return (
        1e0
        / torch.sqrt(torch.tensor(2 * math.pi))
        * torch.exp(-5e-1 * ((x - mean) / sigma) ** 2)
        / sigma
    )


def soft_coulomb(r):
    """Soft Coulomb.

    Evaluates the soft coulomb interaction.
    """
    return 1 / torch.sqrt(r ** 2 + 1e0)


def exp_coulomb(r):
    """Exponential coulomb.

    Evaluates the exponential coulomb interaction.
    """
    return constants.A * torch.exp(-constants.kappa * torch.abs(r))
