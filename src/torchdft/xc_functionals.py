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
    density, A=constants.A, kappa=constants.kappa, thres=1e-15
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

    finite_y = torch.where(y == 0.0, torch.ones(1), y)
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
