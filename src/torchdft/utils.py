# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import math
from typing import Tuple

import torch
from torch import Tensor

from torchdft import constants


class System:
    """System class."""

    def __init__(
        self, nelectrons: int = 0, charges: Tensor = None, centers: Tensor = None
    ):
        self.nelectrons = nelectrons
        self.charges = charges
        self.centers = centers

        assert nelectrons > 0, "nelectrons must be defined in System."
        assert charges is not None, "charges must be defined in System."
        assert centers is not None, "centers must be defined in System."

    def get_occ(self, mode: str = "KS") -> Tensor:
        if mode == "KS":
            n_occ = self.nelectrons // 2 + self.nelectrons % 2
            occ = torch.ones(n_occ)
            occ[: self.nelectrons // 2] += 1
        elif mode == "OF":
            occ = torch.tensor([self.nelectrons])

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

    def eigh(self, A: Tensor) -> Tuple[Tensor, Tensor]:
        w, V = torch.linalg.eigh(self.X.t() @ A @ self.X)
        V = self.X @ V
        return w, V
