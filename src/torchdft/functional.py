# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from abc import abstractmethod

from torch import Tensor

from .density import Density


class Functional:
    """Represents a density functional."""

    requires_grad: bool

    @abstractmethod
    def __call__(self, density: Density) -> Tensor:
        pass