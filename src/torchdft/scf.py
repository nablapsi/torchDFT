# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import torch

from torchdft.functionals import (
    get_external_potential,
    get_hartree_potential,
    get_kinetic_matrix,
    get_XC_energy,
    get_XC_potential,
)
from torchdft.utils import exp_coulomb, get_dx
from torchdft.xc_functionals import exponential_coulomb_LDA_XC_energy_density

__all__ = ["solve_ks"]


def ks_iteration(nelectrons, veff, grid):
    H = get_kinetic_matrix(grid) + torch.diag_embed(veff)
    eigener, eigstates = torch.linalg.eigh(H)
    nocc = nelectrons // 2
    eigener, eigstates = eigener[:nocc], eigstates[:, :nocc]
    dx = get_dx(grid)
    eigstates = eigstates / ((eigstates ** 2).sum(dim=0) * dx).sqrt()  # normalize
    new_density = (2 * (eigstates ** 2)).sum(dim=-1)
    total_eigener = (2 * eigener).sum()
    return new_density, total_eigener


def solve_ks(
    system,
    grid,
    alpha=0.5,
    interaction_fn=exp_coulomb,
    XC_energy_density=exponential_coulomb_LDA_XC_energy_density,
    max_iterations=100,
    print_iterations=False,
):
    """Given a system, evaluates its energy solvig the KS equations."""
    dx = get_dx(grid)

    # Get external potential.
    system = system._replace(
        vext=get_external_potential(
            system.charges, system.centers, grid, interaction_fn
        )
    )

    # First iteration will use vext as veff since we don't have any density.
    new_density, total_eigener = ks_iteration(system.nelectrons, system.vext, grid)
    new_ener = 0e0

    if print_iterations:
        print("Iteration | Old energy / Ha | New energy / Ha | Absolute difference")

    for it in range(1, max_iterations + 1):
        # TODO: Check if this copy is absolutely neccessary.
        old_density = new_density.clone().detach()
        old_ener = new_ener
        v_H = get_hartree_potential(old_density, grid, interaction_fn)
        E_xc = get_XC_energy(old_density, grid, XC_energy_density)
        v_xc = get_XC_potential(old_density, grid, XC_energy_density)
        veff = system.vext + v_H + v_xc
        density, total_eigener = ks_iteration(system.nelectrons, veff, grid)
        new_ener = total_eigener + E_xc - ((v_H / 2 + v_xc) * old_density).sum() * dx
        system = system._replace(density=density)
        system = system._replace(energy=new_ener)

        new_density = old_density + alpha * (density - old_density)

        if print_iterations:
            print(
                "%3i   %10.7f   %10.7f   %3.4e"
                % (it, old_ener, new_ener, torch.abs(old_ener - new_ener))
            )
            # it, old_ener, total_ener)

        # TODO: Add some kind of convergence criteria so not all
        # iterations are evaluated.
        # if torch.abs(old_ener - total_ener) < 1e-5:
        #    break

    return system
