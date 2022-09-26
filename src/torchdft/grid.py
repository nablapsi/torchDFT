# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import torch
from torch import Tensor, nn


class Grid(nn.Module):
    """Base class representing a grid.
    Attributes:
        grid: Tensor holding the grid coordinates.
        grid_weights: Tensor holding the integration weights of grid.
        dv: Volume element. 1 for cartesian grids. 4 pi r^2 for radial grid.
    """

    grid: Tensor
    grid_weights: Tensor
    dv: Tensor


class Uniform1DGrid(Grid):
    """Class to represent a 1D uniform grid."""

    def __init__(
        self,
        end: float,
        dx: float,
        start: float = 0e0,
        reflection_symmetry: bool = False,
    ):
        self.grid = torch.arange(start, end + dx, dx)
        if reflection_symmetry:
            assert start == 0e0
            self.grid = torch.cat((-self.grid[1:].flip(-1), self.grid))
        self.grid_weights = torch.tensor(dx)
        self.dv = torch.tensor(1.0)
