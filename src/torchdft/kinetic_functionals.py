# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import math

from torch import Tensor

from .density import Density
from .functional import Functional


# TODO: TF and vW functionals are written as kinetic energy density, so they don't
# have the usual form found in textbooks. This is because "get_functional_energy" works
# with energy density. It would be nice to evaluate the kinetic functionals in their
# usual way.
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

    def forward(self, density: Density) -> Tensor:
        return self.c * density.value ** 2


class VonWeizsaecker(Functional):
    """Evaluate the von Weizsaecker kinetic energy."""

    requires_grad = True

    def forward(self, density: Density) -> Tensor:
        assert density.grad is not None
        return 1.0 / 8.0 * (density.grad / density.value) ** 2
