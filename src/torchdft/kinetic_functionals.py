# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from .functionals import get_gradient
from .utils import get_dx


def get_TF_energy_1d(density, grid, A=0.3):
    """Evaluate the Thomas Fermi kinetic energy in one dimension.

    https://journals.aps.org/pra/pdf/10.1103/PhysRevA.60.2285
    """
    dx = get_dx(grid)
    return A * (density ** 2).sum() * dx


def get_vW_energy(density, grid):
    """Evaluate the von Weizsaecker kinetic energy."""

    dx = get_dx(grid)
    grid_dim = grid.size(0)
    grad_operator = get_gradient(grid_dim) / dx
    return 1.0 / 8.0 * (grad_operator.mv(density) ** 2 / density).sum() * dx


def get_TF_vW_energy(density, grid):
    return get_TF_energy_1d(density, grid) + get_vW_energy(density, grid)
