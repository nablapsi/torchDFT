# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import math

import torch

from torchdft import constants


def get_LDAX_potential(density, A=constants.A, kappa=constants.kappa, thres=1e-15):
    """Evaluate exchange potential.

    Evaluate exchange potential.
    TODO: I think the equation used in jax_dft for the exchange energy per
    electron (derived from [1]) is not the same as the one given in [2] (this
    one is provided explicitly for the electron per energy).
    I will use the one in [2] multiplied by the prefactor A of the exponential
    coulomb interaction, but this has to be confirmed!!


    [1] https://arxiv.org/pdf/1504.05620.pdf eq 17
    [2] https://arxiv.org/pdf/1408.2434.pdf eq 11
    """

    beta = 2.0 * density * math.pi / kappa
    return torch.where(
        density > thres,
        (A / (2.0 * math.pi))
        * (torch.log(1.0 + beta * beta) / beta - 2 * torch.arctan(beta)),
        (A / (2.0 * math.pi)) * (-beta + beta ** 3 / 6),
    )
