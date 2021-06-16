# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from torch import Tensor

from .density import Density
from .functional import Functional


class ThomasFermi1D(Functional):
    """Evaluate the Thomas Fermi kinetic energy in one dimension.

    https://journals.aps.org/pra/pdf/10.1103/PhysRevA.60.2285
    """

    requires_grad = False

    def __init__(self, A: float = 0.3):
        super().__init__()
        self.A = A

    def forward(self, density: Density) -> Tensor:
        return self.A * density.value ** 2


class VonWeizsaecker(Functional):
    """Evaluate the von Weizsaecker kinetic energy."""

    requires_grad = True

    def forward(self, density: Density) -> Tensor:
        assert density.grad is not None
        return 1.0 / 8.0 * density.grad ** 2 / density.value
