# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .density import Density
from .functional import Functional
from .gridbasis import get_hartree_potential
from .utils import get_dx


class SigLayer(nn.Module):
    """Self Interaction Gate (SIG) layer."""

    def __init__(self, grid: Tensor, interaction_fn: Callable[[Tensor], Tensor]):
        super().__init__()
        self.grid = grid
        self.interaction_fn = interaction_fn
        self.dx = get_dx(self.grid)
        self.sigma = nn.Parameter(torch.Tensor(1))

        # TODO: Check if this should be initialize from another distribution.
        nn.init.uniform_(self.sigma)

    def forward(self, density: Tensor, xc_energy_density: Tensor) -> Tensor:
        density = density.squeeze()
        if not hasattr(self, "Ne"):
            self.Ne = density.detach().sum(-1) * self.dx
        beta = (-(((self.Ne - 1) / self.sigma) ** 2)).exp()[:, None]
        V_H = get_hartree_potential(density, self.grid, self.interaction_fn)
        return xc_energy_density * (1 - beta) - 5e-1 * V_H * beta


class GlobalConvolutionalLayer(nn.Module):
    """Global convolutional layer."""

    g: Tensor

    def __init__(
        self, channels: int, grid: Tensor, minval: float = 0e0, maxval: float = 1e0
    ):
        super().__init__()
        self.channels = channels
        self.dx = get_dx(grid)
        self.register_buffer("g", (grid[:, None] - grid).abs())
        self.maxval = maxval
        self.minval = minval
        self.xi = nn.Parameter(torch.Tensor(self.channels))

        nn.init.uniform_(self.xi)

    def forward(self, density: Tensor) -> Tensor:
        """Forward pass of global convolutional layer.

        Args:
            density: torch tensor of dimension(Nbatch, 1, grid_dim)
            grid: torch tensor of dimension(grid_dim,)
        """
        xi = 1 / (self.minval + (self.maxval - self.minval) * torch.sigmoid(self.xi))
        expo = (-(self.g[None, :, :] * xi[:, None, None])).exp()
        integral = torch.einsum("i, jkl, ilm -> jim", 5e-1 * xi, density, expo)

        return integral * self.dx


class Conv1dPileLayers(nn.Module):
    """Generate a pile of Conv1d layers."""

    def __init__(
        self, channels: List[int], kernels: List[int], negative_transform: bool = True
    ):
        super().__init__()

        assert len(kernels) + 1 == len(channels)
        self.requires_grad = False
        self.transfer = nn.SiLU()
        self.layers = nn.ModuleList()

        for i, (channel, kernel) in enumerate(zip(channels[:-1], kernels)):
            # NOTE: Only integer number kernels can be used to ensure input and
            # output dimensions are kept constant.
            assert kernel % 2 == 1
            padding = (kernel - 1) // 2
            self.layers.append(
                nn.Conv1d(
                    in_channels=channel,
                    out_channels=channels[i + 1],
                    kernel_size=kernel,
                    padding=padding,
                    bias=False,
                )
            )
        self.sign = -1 if negative_transform else 1

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = self.transfer(layer(x))
        return (self.sign * x).squeeze()


class Conv1dFunctionalNet(Functional):
    """Convolutional 1D NN functional.

    Creates LDA and GGA functionals upon the following instantiation.
    LDA: window_size = 1, channels = [1, 16, 16, 16, 1], negative_transform = True
    GGA: window_size = 3, channels = [1, 16, 16, 16, 1], negative_transform = True
    """

    def __init__(
        self, window_size: int, channels: List[int], negative_transform: bool = True
    ):
        super().__init__()

        self.requires_grad = False

        kernels = [1] * (len(channels) - 1)
        kernels[0] = window_size

        self.conv1d = Conv1dPileLayers(channels, kernels, negative_transform)

    def forward(self, den: Density) -> Tensor:
        if len(den.value.shape) == 2:
            x = den.value[:, None, :]
        else:
            x = den.value[None, None, :]

        return self.conv1d(x)


class GlobalFunctionalNet(Functional):
    """Global convolutional 1d NN functional."""

    def __init__(
        self,
        channels: List[int],
        glob_channels: int,
        grid: Tensor,
        kernels: List[int],
        maxval: float = 1e0,
        minval: float = 0e0,
        negative_transform: bool = True,
        sig_layer: bool = True,
        interaction_fn: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        super().__init__()

        self.requires_grad = False

        self.channels = channels
        self.grid = grid
        self.interaction_fn = interaction_fn
        self.kernels = kernels
        self.maxval = maxval
        self.minval = minval
        self.negative_transform = negative_transform
        self.sig_layer = sig_layer

        self.globalconv = GlobalConvolutionalLayer(
            glob_channels, self.grid, maxval=self.maxval, minval=self.minval
        )
        self.conv1d = Conv1dPileLayers(
            kernels=self.kernels,
            channels=self.channels,
            negative_transform=self.negative_transform,
        )
        if self.sig_layer:
            assert self.interaction_fn is not None
            self.sig = SigLayer(self.grid, self.interaction_fn)

    def forward(self, den: Density) -> Tensor:
        if len(den.value.shape) == 2:
            density = den.value[..., None, :]
        else:
            density = den.value[None, None, :]

        x = self.globalconv(density)
        x = self.conv1d(x)

        if self.sig_layer:
            x = self.sig(density, x)

        return x.squeeze()


class GgaConv1dFunctionalNet(Functional):
    """Convolutional 1D NN GGA functional."""

    def __init__(self, channels: List[int], negative_transform: bool = True):
        super().__init__()

        assert channels[0] == 2
        self.requires_grad = True

        kernels = [1] * (len(channels) - 1)

        self.conv1d = Conv1dPileLayers(channels, kernels, negative_transform)

    def forward(self, den: Density) -> Tensor:
        assert den.grad is not None
        x = torch.stack((den.value, den.grad ** 2), dim=-2)
        if len(x.shape) == 2:
            x = x[None, :, :]
        return self.conv1d(x)


class TrainableLDA(Functional):
    """Trainable LDA."""

    requires_grad = False
    per_electron = False

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, density: Density) -> Tensor:
        log_n = density.value.log()
        x = log_n.unsqueeze(dim=-1)
        return -F.softplus(log_n * 4 / 3 + self.mlp(x).squeeze(dim=-1))


class TrainableGGA(Functional):
    """Trainable GGA."""

    requires_grad = True
    per_electron = False

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, density: Density) -> Tensor:
        assert density.grad is not None
        log_n = density.value.log()
        log_grad = density.grad.log()
        x = torch.stack([log_n, log_grad], dim=-1)
        return -F.softplus(log_n * 4 / 3 + self.mlp(x).squeeze(dim=-1))
