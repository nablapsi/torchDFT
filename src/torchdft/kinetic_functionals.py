# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import math

from torch import Tensor

from .density import Density
from .functional import Functional


class ThomasFermi1D(Functional):
    r"""Evaluate the Thomas Fermi kinetic energy density in one dimension.

    N. H. March, S. Kais, Kinetic Energy Functional Derivative for the TF atom
    in D dimension,
    International Journal of Quantum Chemistry, Vol. 65, 411-413 1997

    NOTE: This is NOT the TF functional but the TF energy density. The kinetic
    energy will be evaluated through `get_functional_energy` in `gridbasis` as:
        E = \int ThomasFermi1D()(n) * n * dx
    yielding the actual expression for the TF kinetic energy:
        E = c_k * \int n**3 dx
    """

    def __init__(self, c: float = math.pi ** 2 / 24.0):
        super().__init__()
        self.c = c
        self.requires_grad = False
        self.per_electron = False

    def forward(self, density: Density) -> Tensor:
        return self.c * density.value ** 3


class VonWeizsaecker(Functional):
    """Evaluate the von Weizsaecker kinetic energy."""

    requires_grad = True
    per_electron = False

    def forward(self, density: Density) -> Tensor:
        assert density.grad is not None
        return 1.0 / 8.0 * density.grad ** 2 / density.value
