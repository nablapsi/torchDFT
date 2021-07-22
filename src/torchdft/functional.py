# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from torch import Tensor, nn

from .density import Density


class Functional(nn.Module, ABC):
    """Represents a density functional."""

    requires_grad: bool

    @abstractmethod
    def forward(self, density: Density) -> Tensor:
        pass


class ComposedFunctional(Functional):
    """Linear combination of density functionals."""

    def __init__(
        self, functionals: List[Functional], factors: Optional[List[float]] = None
    ):
        super().__init__()
        self.functionals = nn.ModuleList(functionals)
        self.factors = factors if factors is not None else [1] * len(functionals)
        self.requires_grad = any(functional.requires_grad for functional in functionals)

    def forward(self, density: Density) -> Tensor:
        epsilon = torch.stack(
            [
                factor * functional(density)
                for (functional, factor) in zip(self.functionals, self.factors)
            ],
            dim=-1,
        ).sum(-1)
        return epsilon
