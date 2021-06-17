# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Tuple, Union

import torch
from torch import Tensor

from .basis import Basis
from .errors import SCFNotConverged
from .functional import Functional
from .gridbasis import GridBasis
from .utils import GeneralizedDiagonalizer

__all__ = ["solve_scf"]


def ks_iteration(F: Tensor, X: Tensor, occ: Tensor) -> Tuple[Tensor, Tensor]:
    n_occ = occ.shape[-1]
    epsilon, C = GeneralizedDiagonalizer.eigh(F, X)
    epsilon, C = epsilon[..., :n_occ], C[..., :n_occ]
    P = (C * occ[..., None, :]) @ C.transpose(-2, -1)
    energy_orb = (epsilon * occ).sum(-1)
    return P, energy_orb


def solve_scf(
    basis: Basis,
    occ: Tensor,
    xc_functional: Functional,
    alpha: float = 0.5,
    alpha_decay: float = 1.0,
    max_iterations: int = 100,
    density_threshold: float = 1e-5,
    print_iterations: Union[bool, int] = False,
    mode: str = "KS",
    enforce_symmetry: bool = False,
    create_graph: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Given a system, evaluates its energy by solving the KS equations."""
    S, T, V_ext = basis.get_core_integrals()
    S = GeneralizedDiagonalizer(S)
    F = T + V_ext
    P_in, energy_orb = ks_iteration(F, S.X, occ)
    if enforce_symmetry and isinstance(basis, GridBasis):
        P_in = basis.symmetrize_P(P_in)
    energy_prev = energy_orb + basis.E_nuc
    print_iterations = print_iterations and len(P_in.shape) == 2
    if print_iterations:
        print("Iteration | Old energy / Ha | New energy / Ha | Density diff norm")
    for i in range(max_iterations):
        V_H, V_xc, E_xc = basis.get_int_integrals(
            P_in, xc_functional, create_graph=create_graph
        )
        F = T + V_ext + V_H + V_xc
        P_out, energy_orb = ks_iteration(F, S.X, occ)
        # TODO duplicate, should be made part of ks_iteration()
        if enforce_symmetry and isinstance(basis, GridBasis):
            P_out = basis.symmetrize_P(P_out)
        energy = (
            energy_orb + E_xc - ((V_H / 2 + V_xc) * P_in).sum((-2, -1)) + basis.E_nuc
        )
        density_diff = basis.density_mse(basis.density(P_out - P_in))
        if print_iterations and i % print_iterations == 0:
            print(
                "%3i   %10.7f   %10.7f   %3.4e" % (i, energy_prev, energy, density_diff)
            )
        converged = density_diff < density_threshold
        if converged.all():
            break
        alpha_masked = torch.where(converged, 0.0, alpha)[..., None, None]
        P_in = P_in + alpha_masked * (P_out - P_in)
        alpha = alpha * alpha_decay
        energy_prev = energy
    else:
        raise SCFNotConverged()
    return P_out.detach(), energy.detach()
