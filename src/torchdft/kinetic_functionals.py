# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from .utils import get_dx


def TF_energy_1d(density, grid, A=0.3):
    """Evaluate the Thomas Fermi kinetic energy in one dimension.

    https://journals.aps.org/pra/pdf/10.1103/PhysRevA.60.2285
    """
    dx = get_dx(grid)
    return A * (density.value ** 2).sum() * dx


def vW_energy(density, grid):
    """Evaluate the von Weizsaecker kinetic energy."""

    dx = get_dx(grid)
    return 1.0 / 8.0 * (density.grad ** 2 / density.value).sum() * dx
