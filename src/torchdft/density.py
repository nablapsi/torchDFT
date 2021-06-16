# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from dataclasses import dataclass
from typing import Optional

from torch import Tensor


@dataclass
class Density:
    """Density data structure."""

    value: Tensor
    grad: Optional[Tensor] = None
