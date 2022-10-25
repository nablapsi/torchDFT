# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import List

import torch
from torch import Tensor, nn


class Grid(nn.Module):
    """Base class representing a grid.

    Attributes:
        nodes: Tensor holding the grid coordinates.
        grid_weights: Tensor holding the integration weights of grid.
        dv: Volume element. 1 for cartesian grids. 4 pi r^2 for radial grid.
    """

    nodes: Tensor
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
        self.nodes = torch.arange(start, end + dx, dx)
        if reflection_symmetry:
            assert start == 0e0
            self.nodes = torch.cat((-self.nodes[1:].flip(-1), self.nodes))
        self.grid_weights = torch.tensor([dx])
        self.dv = torch.tensor([1.0])


class RadialGrid(Grid):
    """Class to represent a radial grid."""

    def __init__(
        self,
        end: float,
        dx: float,
    ):
        self.nodes = torch.arange(dx, end + dx, dx)
        self.grid_weights = torch.tensor([dx])
        self.dv = 4 * torch.pi * self.nodes**2


class GridBatch(Grid):
    """Class to deal with grids of different dimensions."""

    def __init__(self, grid_list: List[Grid]):
        n_grids = len(grid_list)
        max_grid_len = max([len(grid.nodes) for grid in grid_list])
        max_gridw_len = max([len(grid.grid_weights) for grid in grid_list])
        max_dv_len = max([len(grid.dv) for grid in grid_list])
        self.nodes = grid_list[0].nodes.new_zeros(n_grids, max_grid_len)
        self.grid_weights = grid_list[0].nodes.new_zeros(n_grids, max_gridw_len)
        self.dv = grid_list[0].nodes.new_zeros(n_grids, max_dv_len)
        for i, g in enumerate(grid_list):
            maxg = g.nodes.shape[0]
            maxgw = max_gridw_len if max_gridw_len == 1 else g.grid_weights.shape[0]
            maxdv = max_dv_len if max_dv_len == 1 else g.dv.shape[0]
            self.nodes[i, :maxg] = g.nodes
            self.grid_weights[i, :maxgw] = g.grid_weights
            self.dv[i, :maxdv] = g.dv
        self.grid_weights = self.grid_weights.squeeze(-1)
        self.dv = self.dv.squeeze(-1)
