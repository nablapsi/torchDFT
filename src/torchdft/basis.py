# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from abc import ABC, abstractmethod
from typing import Dict, Tuple

from torch import Tensor, nn

from .functional import Functional


class Basis(nn.Module, ABC):
    """Base class representing an abstract basis."""

    E_nuc: Tensor

    @abstractmethod
    def get_core_integrals(self) -> Tuple[Tensor, Tensor, Tensor]:
        pass

    @abstractmethod
    def get_int_integrals(
        self, P: Tensor, functional: Functional, create_graph: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        pass

    @abstractmethod
    def density_mse(self, density: Tensor) -> Tensor:
        pass

    @abstractmethod
    def density(self, P: Tensor) -> Tensor:
        pass

    def density_metrics_fn(
        self, density: Tensor, density_ref: Tensor
    ) -> Dict[str, Tensor]:
        return {}
