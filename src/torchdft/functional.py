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
    per_electron: bool = True

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
        self.per_electron = all(functional.per_electron for functional in functionals)

    def forward(self, density: Density) -> Tensor:
        eps = density.value.new_zeros(density.density.shape)
        for factor, functional in zip(self.factors, self.functionals):
            eps_func = factor * functional(density)
            if not self.per_electron and functional.per_electron:
                eps_func = eps_func * density.density
            eps = eps + eps_func
        return eps


class NullFunctional(Functional):
    """Null functional for non interacting calculations."""

    requires_grad: bool = False
    per_electron: bool = True

    def forward(self, density: Density) -> Tensor:
        return torch.zeros_like(density.value)
