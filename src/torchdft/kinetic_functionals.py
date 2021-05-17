# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


class ThomasFermi1D:
    """Evaluate the Thomas Fermi kinetic energy in one dimension.

    https://journals.aps.org/pra/pdf/10.1103/PhysRevA.60.2285
    """

    requires_grad = False

    def __init__(self, A=0.3):
        self.A = A

    def __call__(self, density):
        return self.A * density.value ** 2


class VonWeizsaecker:
    """Evaluate the von Weizsaecker kinetic energy."""

    requires_grad = True

    def __call__(self, density):
        return 1.0 / 8.0 * density.grad ** 2 / density.value
