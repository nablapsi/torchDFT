# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from .utils import GeneralizedDiagonalizer
from .xc_functionals import exponential_coulomb_LDA_XC_energy_density

__all__ = ["solve_ks"]


def ks_iteration(F, S, n_electrons):
    n_occ = n_electrons // 2 + n_electrons % 2
    occ = F.new_ones(n_occ)  # orbital occupation numbers
    occ[: n_electrons // 2] += 1
    epsilon, C = S.eigh(F)
    epsilon, C = epsilon[:n_occ], C[:, :n_occ]
    P = (C * occ) @ C.t()
    energy_orb = (epsilon * occ).sum()
    return P, energy_orb


def solve_ks(
    system,
    basis,
    alpha=0.5,
    XC_energy_density=exponential_coulomb_LDA_XC_energy_density,
    max_iterations=100,
    dm_threshold=1e-3,
    print_iterations=False,
):
    """Given a system, evaluates its energy by solving the KS equations."""
    S, T, V_ext = basis.get_core_integrals()
    S = GeneralizedDiagonalizer(S)
    F = T + V_ext
    P_in, energy_prev = ks_iteration(F, S, system.nelectrons)
    if print_iterations:
        print("Iteration | Old energy / Ha | New energy / Ha | Absolute difference")
    for i in range(max_iterations):
        V_H, V_xc, E_xc = basis.get_int_integrals(P_in, XC_energy_density)
        F = T + V_ext + V_H + V_xc
        P_out, energy_orb = ks_iteration(F, S, system.nelectrons)
        energy = energy_orb + E_xc - ((V_H / 2 + V_xc) * P_in).sum()
        if print_iterations:
            print(
                "%3i   %10.7f   %10.7f   %3.4e"
                % (i, energy_prev, energy, (energy - energy_prev).abs())
            )
        if (P_out - P_in).norm() < dm_threshold:
            break
        P_in = P_in + alpha * (P_out - P_in)
        energy_prev = energy
    return P_out, energy
