# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import math

import torch

from torchdft import constants


def exponential_coulomb_LDA_XC_energy_density(density):
    """LDA XC energy for exponential coulomb interaction."""
    return get_exponential_coulomb_LDAX_energy_density(
        density
    ) + get_exponential_coulomb_LDAC_energy_density(density)


def get_exponential_coulomb_LDAX_energy_density(
    density, A=constants.A, kappa=constants.kappa, thres=1e-15
):
    """Evaluate exchange potential.

    Evaluate exchange potential.
    [1] https://arxiv.org/pdf/1504.05620.pdf eq 17
    """

    y = density * math.pi / kappa
    return torch.where(
        density > thres,
        (A / (2.0 * math.pi)) * (torch.log(1.0 + y * y) / y - 2 * torch.arctan(y)),
        (A / (2.0 * math.pi)) * (-y + y ** 3 / 6),
    )


def get_exponential_coulomb_LDAC_energy_density(
    density, A=constants.A, kappa=constants.kappa
):
    """Evaluate correlation potential.

    Evaluate exchange potential.
    """
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


def lda_pw92(density):
    """Perdew--Wang 1992 parametrization of LDA."""

    def r_s(n):
        return (3 / (4 * math.pi * n)) ** (1 / 3)

    def k_F(n):
        return (3 * math.pi ** 2 * n) ** (1 / 3)

    def epsilon_Xunif(n):
        return -3 * k_F(n) / (4 * math.pi)

    def epsilon_Cunif(rs, zeta):
        return (
            epsilon_c0(rs)
            + alpha_c(rs) * ff(zeta) / ff0 * (1 - zeta ** 4)
            + (epsilon_c1(rs) - epsilon_c0(rs)) * ff(zeta) * zeta ** 4
        )

    def ff(zeta):
        return ((1 + zeta) ** (4 / 3) + (1 - zeta) ** (4 / 3) - 2) / (2 ** (4 / 3) - 2)

    ff0 = 1.709921

    def epsilon_c0(rs):
        return Gamma(rs, 0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, 1)

    def epsilon_c1(rs):
        return Gamma(rs, 0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517, 1)

    def alpha_c(rs):
        return -Gamma(rs, 0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671, 1)

    def Gamma(rs, A, a1, b1, b2, b3, b4, p):
        poly = b1 * rs ** (1 / 2) + b2 * rs + b3 * rs ** (3 / 2) + b4 * rs ** (p + 1)
        return -2 * A * (1 + a1 * rs) * torch.log(1 + 1 / (2 * A * poly))

    return epsilon_Xunif(density) + epsilon_Cunif(r_s(density), 0)
