# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class Density:
    """Density data structure."""

    value: Tensor
    grid: Tensor
    grid_weights: Tensor
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
            self.value <= 1e-100,
            self.value.new_full((1,), 1e-100),
            self.value,
        )
