# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


class Density:
    """Density data structure."""

    def __init__(self, density, grad=None):
        self.value = density
        self.grad = grad

    def detach(self):
        value = self.value.detach()

        if self.grad is not None:
            grad = self.grad.detach()
        else:
            grad = None
        return Density(value, grad)
