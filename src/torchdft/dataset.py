# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .utils import System


def collate_fn(
    batch: List[Tuple[System, Tensor, Tensor]]
) -> Tuple[List[System], Tensor, Tensor]:
    """Get a batch of data and adds an outer dimension."""
    sample = list(zip(*batch))
    # TODO: Add case for gaussian basis.
    if isinstance(sample[0][0], System):
        systems = list(sample[0])
    energies = torch.stack(sample[1])
    densities = torch.stack(sample[2])
    return systems, energies, densities


class SystemDataSet(Dataset[Tuple[System, Tensor, Tensor]]):
    """Data loader for DFT functional training."""

    def __init__(self, systems: List[System], energies: Tensor, densities: Tensor):
        assert len(systems) == energies.shape[0] == densities.shape[0]
        self.systems = systems
        self.energies = energies
        self.densities = densities

    def __len__(self) -> int:
        return len(self.systems)

    def __getitem__(self, idx: int) -> Tuple[System, Tensor, Tensor]:
        return self.systems[idx], self.energies[idx], self.densities[idx]
