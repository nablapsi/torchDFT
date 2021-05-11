# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import torch

from .functionals import (
    get_external_potential,
    get_hartree_potential,
    get_kinetic_matrix,
    get_XC_energy,
    get_XC_potential,
)
from .utils import GeneralizedDiagonalizer, exp_coulomb, get_dx
from .xc_functionals import exponential_coulomb_LDA_XC_energy_density

__all__ = ["solve_ks"]


def ks_iteration(F, S, n_electrons):
    n_occ = n_electrons // 2 + n_electrons % 2
    epsilon, phi = S.eigh(F)
    epsilon, phi = epsilon[:n_occ], phi[:, :n_occ]
    density = (torch.column_stack((phi ** 2,) * 2)[:, :n_electrons]).sum(dim=-1)
    energy_orb = ((torch.column_stack((epsilon,) * 2)[:n_electrons])).sum()
    return density, energy_orb


def solve_ks(
    system,
    grid,
    alpha=0.5,
    interaction_fn=exp_coulomb,
    XC_energy_density=exponential_coulomb_LDA_XC_energy_density,
    max_iterations=100,
    density_threshold=1e-4,
    print_iterations=False,
):
    """Given a system, evaluates its energy by solving the KS equations."""
    dx = get_dx(grid)
    S = torch.full((len(grid),), dx).diag_embed()  # overlap matrix
    S = GeneralizedDiagonalizer(S)
    T = dx * get_kinetic_matrix(grid)
    v_ext = dx * get_external_potential(
        system.charges, system.centers, grid, interaction_fn
    )
    F = T + v_ext.diag_embed()
    density_in, energy_prev = ks_iteration(F, S, system.nelectrons)
    if print_iterations:
        print("Iteration | Old energy / Ha | New energy / Ha | Absolute difference")
    for i in range(max_iterations):
        v_H = dx * get_hartree_potential(density_in, grid, interaction_fn)
        E_xc = get_XC_energy(density_in, grid, XC_energy_density)
        v_xc = dx * get_XC_potential(density_in, grid, XC_energy_density)
        v_eff = v_ext + v_H + v_xc
        F = T + v_eff.diag_embed()
        density_out, energy_orb = ks_iteration(F, S, system.nelectrons)
        energy = energy_orb + E_xc - ((v_H / 2 + v_xc) * density_in).sum()
        if print_iterations:
            print(
                "%3i   %10.7f   %10.7f   %3.4e"
                % (i, energy_prev, energy, (energy - energy_prev).abs())
            )
        if (density_out - density_in).abs().sum() * dx < density_threshold:
            break
        density_in = density_in + alpha * (density_out - density_in)
        energy_prev = energy
    return density_out, energy
