# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import math

import torch

from torchdft import constants

from .functional import Functional


class Lda1d(Functional):
    """LDA XC functional in 1D."""

    requires_grad = False

    def __call__(self, density):
        """LDA XC energy for exponential coulomb interaction."""
        return self.get_exponential_coulomb_LDAX_energy_density(
            density
        ) + self.get_exponential_coulomb_LDAC_energy_density(density)

    def get_exponential_coulomb_LDAX_energy_density(
        self, density, A=constants.A, kappa=constants.kappa, thres=1e-15
    ):
        """Evaluate exchange potential.

        Evaluate exchange potential.
        [1] https://arxiv.org/pdf/1504.05620.pdf eq 17
        """

        density = density.value
        y = density * math.pi / kappa
        return torch.where(
            density > thres,
            (A / (2.0 * math.pi)) * (torch.log(1.0 + y * y) / y - 2 * torch.arctan(y)),
            (A / (2.0 * math.pi)) * (-y + y ** 3 / 6),
        )

    def get_exponential_coulomb_LDAC_energy_density(
        self, density, A=constants.A, kappa=constants.kappa
    ):
        """Evaluate correlation potential.

        Evaluate exchange potential.
        """
        density = density.value
        y = density * math.pi / kappa
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
            / math.pi
            / (
                alpha
                + beta * torch.sqrt(finite_y)
                + gamma * finite_y
                + delta * finite_y ** 1.5
                + eta * finite_y ** 2
                + sigma * finite_y ** 2.5
                + nu * math.pi * kappa ** 2 / A * finite_y ** 3
            )
        )
        return torch.where(y == 0.0, -A * y / math.pi / alpha, out)


class LdaPw92(Functional):
    """Perdew--Wang 1992 parametrization of LDA."""

    requires_grad = False

    def __call__(self, density):
        eps_x, eps_c, *_ = _lda_pw92(density.value)
        return eps_x + eps_c


def _lda_pw92(density):
    def Gamma(A, a1, b1, b2, b3, b4, p):
        poly = b1 * rs ** (1 / 2) + b2 * rs + b3 * rs ** (3 / 2) + b4 * rs ** (p + 1)
        return -2 * A * (1 + a1 * rs) * torch.log(1 + 1 / (2 * A * poly))

    zeta = torch.tensor(0)
    rs = (3 / (4 * math.pi * density)) ** (1 / 3)
    kF = (3 * math.pi ** 2 * density) ** (1 / 3)
    eps_x = -3 * kF / (4 * math.pi)
    ff0 = 1.709921
    ff = ((1 + zeta) ** (4 / 3) + (1 - zeta) ** (4 / 3) - 2) / (2 ** (4 / 3) - 2)
    eps_c0 = Gamma(0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, 1)
    eps_c1 = Gamma(0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517, 1)
    alpha_c = -Gamma(0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671, 1)
    eps_c = (
        eps_c0
        + alpha_c * ff / ff0 * (1 - zeta ** 4)
        + (eps_c1 - eps_c0) * ff * zeta ** 4
    )
    return eps_x, eps_c, kF, zeta


class PBE(Functional):
    """Perdew--Burke--Ernzerhof functional."""

    requires_grad = True

    def __call__(self, density):
        eps_x, eps_c, kF, zeta = _lda_pw92(density.value)
        s = density.grad / (2 * kF * density.value)
        ks = torch.sqrt(4 * kF / math.pi)
        phi = ((1 + zeta) ** (2 / 3) + (1 - zeta) ** (2 / 3)) / 2
        t = density.grad / (2 * phi * ks * density.value)
        beta = 0.066725
        kappa = 0.804
        mu = beta * (math.pi ** 2 / 3)
        FX = 1 + kappa - kappa / (1 + mu * s ** 2 / kappa)
        eps_x = eps_x * FX
        gamma = (1 - math.log(2)) / math.pi ** 2
        A = beta / gamma * 1 / (torch.exp(-eps_c / (gamma * phi ** 3)) - 1 + 1e-30)
        poly = t ** 2 * (1 + A * t ** 2) / (1 + A * t ** 2 + A ** 2 * t ** 4)
        H = gamma * phi ** 3 * torch.log(1 + beta / gamma * poly)
        eps_c = eps_c + H
        return eps_x + eps_c
