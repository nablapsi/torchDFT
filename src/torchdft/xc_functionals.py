# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import math
from typing import Tuple

import torch
from torch import Tensor

from torchdft import constants

from .density import Density
from .functional import Functional


class Lda1d(Functional):
    """LDA XC functional in 1D."""

    requires_grad = False

    def forward(self, density: Density) -> Tensor:
        """LDA XC energy for exponential coulomb interaction."""
        return self.get_exponential_coulomb_LDAX_energy_density(
            density
        ) + self.get_exponential_coulomb_LDAC_energy_density(density)

    def get_exponential_coulomb_LDAX_energy_density(
        self,
        density: Density,
        A: float = constants.A,
        kappa: float = constants.kappa,
        thres: float = 1e-15,
    ) -> Tensor:
        """Evaluate exchange potential.

        Evaluate exchange potential.
        [1] https://arxiv.org/pdf/1504.05620.pdf eq 17
        """

        density = density.value
        y = density * torch.pi / kappa
        return torch.where(
            density > thres,
            (A / (2.0 * torch.pi)) * (torch.log(1.0 + y * y) / y - 2 * torch.arctan(y)),
            (A / (2.0 * torch.pi)) * (-y + y**3 / 6),
        )

    def get_exponential_coulomb_LDAC_energy_density(
        self, density: Density, A: float = constants.A, kappa: float = constants.kappa
    ) -> Tensor:
        """Evaluate correlation potential.

        Evaluate exchange potential.
        """
        density = density.value
        y = density * torch.pi / kappa
        alpha = 2.0
        beta = -1.00077
        gamma = 6.26099
        delta = -11.9041
        eta = 9.62614
        sigma = -1.48334
        nu = 1.0

        finite_y = torch.where(y == 0.0, density.new_tensor(1), y)
        out = (
            -A
            * finite_y
            / torch.pi
            / (
                alpha
                + beta * torch.sqrt(finite_y)
                + gamma * finite_y
                + delta * finite_y**1.5
                + eta * finite_y**2
                + sigma * finite_y**2.5
                + nu * torch.pi * kappa**2 / A * finite_y**3
            )
        )
        return torch.where(y == 0.0, -A * y / torch.pi / alpha, out)


class LdaPw92(Functional):
    """Perdew--Wang 1992 parametrization of LDA."""

    requires_grad = False

    def forward(self, density: Density) -> Tensor:
        eps_x, eps_c, *_ = _lda_pw92(density.nu, density.nd)
        return eps_x + eps_c


def _lda_pw92(nu: Tensor, nd: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    def Gamma(
        A: float, a1: float, b1: float, b2: float, b3: float, b4: float, p: int
    ) -> Tensor:
        poly = b1 * rs ** (1 / 2) + b2 * rs + b3 * rs ** (3 / 2) + b4 * rs ** (p + 1)
        return -2 * A * (1 + a1 * rs) * torch.log(1 + 1 / (2 * A * poly))

    def lda(n: Tensor) -> Tensor:
        kF = (3 * torch.pi**2 * n) ** (1 / 3)
        return -3 * kF / (4 * torch.pi)

    density = nu + nd
    zeta = (nu - nd) / density
    rs = (3 / (4 * torch.pi * density)) ** (1 / 3)
    kF = (3 * torch.pi**2 * density) ** (1 / 3)
    eps_x = (
        -3
        / (4 * torch.pi * rs)
        * (9 * torch.pi / 4) ** (1 / 3)
        * ((1 + zeta) ** (4 / 3) + (1 - zeta) ** (4 / 3))
        / 2
    )
    ff0 = 1.709921
    ff = ((1 + zeta) ** (4 / 3) + (1 - zeta) ** (4 / 3) - 2) / (2 ** (4 / 3) - 2)
    eps_c0 = Gamma(0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, 1)
    eps_c1 = Gamma(0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517, 1)
    alpha_c = -Gamma(0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671, 1)
    eps_c = (
        eps_c0
        + alpha_c * ff / ff0 * (1 - zeta**4)
        + (eps_c1 - eps_c0) * ff * zeta**4
    )
    return eps_x, eps_c, kF, zeta


class PBE(Functional):
    """Perdew--Burke--Ernzerhof functional."""

    requires_grad = True

    def forward(self, density: Density) -> Tensor:
        eps_x, eps_c, kF, zeta = _lda_pw92(density.nu, density.nd)
        s = density.grad / (2 * kF * density.density)
        ks = torch.sqrt(4 * kF / torch.pi)
        phi = ((1 + zeta) ** (2 / 3) + (1 - zeta) ** (2 / 3)) / 2
        t = density.grad / (2 * phi * ks * density.density)
        beta = 0.066725
        kappa = 0.804
        mu = beta * (torch.pi**2 / 3)
        FX = 1 + kappa - kappa / (1 + mu * s**2 / kappa)
        eps_x = eps_x * FX
        gamma = (1 - math.log(2)) / torch.pi**2
        A = beta / gamma * 1 / (torch.exp(-eps_c / (gamma * phi**3)) - 1 + 1e-30)
        poly = t**2 * (1 + A * t**2) / (1 + A * t**2 + A**2 * t**4)
        H = gamma * phi**3 * torch.log(1 + beta / gamma * poly)
        eps_c = eps_c + H
        return eps_x + eps_c
