# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import torch

from .utils import GeneralizedDiagonalizer
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
    basis,
    alpha=0.5,
    XC_energy_density=exponential_coulomb_LDA_XC_energy_density,
    max_iterations=100,
    density_threshold=1e-4,
    print_iterations=False,
):
    """Given a system, evaluates its energy by solving the KS equations."""
    S, T, v_ext = basis.get_core_integrals()
    S = GeneralizedDiagonalizer(S)
    F = T + v_ext.diag_embed()
    density_in, energy_prev = ks_iteration(F, S, system.nelectrons)
    if print_iterations:
        print("Iteration | Old energy / Ha | New energy / Ha | Absolute difference")
    for i in range(max_iterations):
        v_H, v_xc, E_xc = basis.get_int_integrals(density_in, XC_energy_density)
        v_eff = v_ext + v_H + v_xc
        F = T + v_eff.diag_embed()
        density_out, energy_orb = ks_iteration(F, S, system.nelectrons)
        energy = energy_orb + E_xc - ((v_H / 2 + v_xc) * density_in).sum()
        if print_iterations:
            print(
                "%3i   %10.7f   %10.7f   %3.4e"
                % (i, energy_prev, energy, (energy - energy_prev).abs())
            )
        if basis.integrate((density_out - density_in).abs()) < density_threshold:
            break
        density_in = density_in + alpha * (density_out - density_in)
        energy_prev = energy
    return density_out, energy
