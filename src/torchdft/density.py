# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Optional

from torch import Tensor


class Density:
    """Density data structure."""

    def __init__(self, density: Tensor, grad: Tensor = None):
        self.value = density
        self.grad = grad
