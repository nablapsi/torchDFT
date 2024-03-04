# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class Density:
    """Density data structure.

    Attributes:
        - density: Total density. Sum of spin up and down densities.
            shape = [batch, basis].
        - value: Total density in spin unpolarized systems and up/down densities
            in spin polarized systems. shape = [batch, (spin), basis].
        - nu, nd: Up and down densities. shape = [batch, basis]
    """

    density: Tensor
    value: Tensor
    grid: Tensor
    grid_weights: Tensor
    nd: Tensor
    nu: Tensor
    grad: Optional[Tensor] = None

    def __init__(
        self,
        value: Tensor,
        grid: Tensor,
        grid_weights: Tensor,
        grad: Optional[Tensor] = None,
    ):
        self.value = value
        self.grid = grid
        self.grid_weights = grid_weights
        self.grad = grad
        self.value = torch.where(
            self.value <= 1e-20,
            self.value.new_full((1,), 1e-20),
            self.value,
        )
        if len(self.value.shape) == 3:
            self.nu, self.nd = self.value[..., 0, :], self.value[..., 1, :]
        else:
            self.nu = self.nd = self.value / 2
        self.density = self.nu + self.nd
