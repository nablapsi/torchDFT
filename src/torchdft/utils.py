# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor

from torchdft import constants


@dataclass
class System:
    """Represents an electronic system."""

    n_electrons: int
    charges: Tensor
    centers: Tensor

    def occ(self, mode: str = "KS") -> Tensor:
        if mode == "KS":
            n_occ = self.n_electrons // 2 + self.n_electrons % 2
            occ = self.centers.new_ones([n_occ])
            occ[: self.n_electrons // 2] += 1
        elif mode == "OF":
            occ = self.centers.new_tensor([self.n_electrons])
        return occ


class SystemBatch:
    """Hold a batch of systems."""

    def __init__(self, systems: List[System]):
        self.systems = systems
        self.nbatch = len(systems)
        self.max_centers = 0
        self.n_electrons = []
        for system in systems:
            center_dim = system.centers.shape[0]
            if center_dim > self.max_centers:
                self.max_centers = center_dim

            self.n_electrons.append(system.n_electrons)

        self.centers = self.systems[0].centers.new_zeros(self.nbatch, self.max_centers)
        self.charges = self.systems[0].centers.new_zeros(self.nbatch, self.max_centers)
        self.n_electrons = self.systems[0].centers.new_tensor(
            self.n_electrons, dtype=torch.uint8
        )

        for i, system in enumerate(systems):
            self.centers[i, : system.centers.shape[0]] = system.centers
            self.charges[i, : system.centers.shape[0]] = system.charges

    def occ(self, mode: str = "KS") -> Tensor:
        if mode == "KS":
            double_occ = self.n_electrons.div(2, rounding_mode="floor")
            n_occ = double_occ + self.n_electrons % 2
            occ = self.centers.new_zeros(self.nbatch, n_occ.max().item())
            for i in range(self.nbatch):
                occ[i, : n_occ[i]] = 1
                occ[i, : double_occ[i]] += 1
        elif mode == "OF":
            occ = self.n_electrons[:, None]
        return occ


def get_dx(grid: Tensor) -> float:
    """Get grid spacing.

    Given a grid as a 1D array returns the spacing between grid points.
    Args:
        grid: Float torch array of dimension (grid_dim,).

    Returns:
        Float.
    """
    grid_dim = grid.size(0)
    return ((torch.amax(grid) - torch.amin(grid)) / (grid_dim - 1)).item()


def gaussian(x: Tensor, mean: float, sigma: float) -> Tensor:
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


def soft_coulomb(r: Tensor) -> Tensor:
    """Soft Coulomb.

    Evaluates the soft coulomb interaction.
    """
    return 1 / torch.sqrt(r ** 2 + 1e0)


def exp_coulomb(r: Tensor) -> Tensor:
    """Exponential coulomb.

    Evaluates the exponential coulomb interaction.
    """
    return constants.A * torch.exp(-constants.kappa * torch.abs(r))


class GeneralizedDiagonalizer:
    """Solves the generalized eigenvalue problem A x = a B x."""

    def __init__(self, B: Tensor):
        B, U = torch.linalg.eigh(B)
        self.X = U @ (1 / B.sqrt()).diag_embed()

    @staticmethod
    def eigh(A: Tensor, X: Tensor) -> Tuple[Tensor, Tensor]:
        w, V = torch.linalg.eigh(X.t() @ A @ X)
        V = X @ V
        return w, V

    def __call__(self, A: Tensor) -> Tuple[Tensor, Tensor]:
        return self.eigh(A, self.X)
