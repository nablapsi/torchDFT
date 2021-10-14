# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Any, Dict, Iterable, List, Tuple, Union

import torch
import xitorch
import xitorch.linalg
from torch import Tensor

from .basis import Basis
from .errors import SCFNotConvergedError
from .functional import Functional
from .gridbasis import GridBasis
from .utils import GeneralizedDiagonalizer, orthogonalizer

__all__ = ["solve_scf"]
DEFAULT_MIXER = "linear"


def ks_iteration(
    F: Tensor, S: Tensor, occ: Tensor, use_xitorch: bool = False
) -> Tuple[Tensor, Tensor]:
    n_occ = occ.shape[-1]
    if use_xitorch:
        F, S = (xitorch.LinearOperator.m(x, is_hermitian=True) for x in [F, S])
        epsilon, C = xitorch.linalg.symeig(F, n_occ, "lowest", S)
    else:
        epsilon, C = GeneralizedDiagonalizer.eigh(F, S)
        epsilon, C = epsilon[..., :n_occ], C[..., :n_occ]
    P = (C * occ[..., None, :]) @ C.transpose(-2, -1)
    energy_orb = (epsilon * occ).sum(dim=-1)
    return P, energy_orb


class DIIS:
    def __init__(
        self,
        max_history: int = 10,
        precondition: bool = True,
        regularization: float = 1e-4,
    ) -> None:
        assert not (regularization and not precondition)
        self.max_history = max_history
        self.precondition = precondition
        self.regularization = regularization
        self.history: List[Tuple[Tensor, Tensor]] = []

    def _get_coeffs(self, X: Tensor, err: Tensor) -> Tensor:
        self.history.append((X, err))
        self.history = self.history[-self.max_history :]
        err = torch.stack([e for _, e in self.history], dim=-3)
        derr = err.diff(dim=-3)
        B = torch.einsum("...imn,...jmn->...ij", derr, derr)
        y = -torch.einsum("...imn,...mn->...i", derr, err[..., -1, :, :])
        *nb, N = B.shape[:-1]
        if N == 0:
            return B.new_ones(*nb, 1)
        if self.precondition:
            pre = 1 / B.detach().diagonal(dim1=-1, dim2=-2).sqrt()
        else:
            pre = B.new_ones((*nb, N))
        B = pre[..., None] * pre[..., None, :] * B
        B = B + self.regularization * torch.eye(B.shape[-1], device=B.device)
        c = pre * torch.linalg.solve(B, pre * y)
        c = torch.cat([-c[..., :1], -c.diff(dim=-1), 1 + c[..., -1:]], dim=-1)
        return c

    def step(self, X: Tensor, err: Tensor, alpha: float = None) -> Tensor:
        c = self._get_coeffs(X, err)
        X = torch.stack([X for X, _ in self.history], dim=-1)
        err = torch.stack([e for _, e in self.history], dim=-1)
        if alpha is not None:
            X = X + alpha * err
        X = (c[..., None, None, :] * X).sum(dim=-1)
        return X


def solve_scf(  # noqa: C901 TODO too complex
    basis: Basis,
    occ: Tensor,
    functional: Functional,
    alpha: float = 0.5,
    alpha_decay: float = 1.0,
    max_iterations: int = 100,
    iterations: Iterable[int] = None,
    density_threshold: float = 1e-4,
    print_iterations: Union[bool, int] = False,
    enforce_symmetry: bool = False,
    tape: List[Tuple[Tensor, Tensor]] = None,
    create_graph: bool = False,
    use_xitorch: bool = True,
    mixer: str = None,
    P_guess: Tensor = None,
    mixer_kwargs: Dict[str, Any] = None,
) -> Tuple[Tensor, Tensor]:
    """Given a system, evaluates its energy by solving the KS equations."""
    mixer = mixer or DEFAULT_MIXER
    assert mixer in {"linear", "pulay", "pulaydensity"}
    S, T, V_ext = basis.get_core_integrals()
    if mixer in {"pulay", "pulaydensity"}:
        diis = DIIS(**(mixer_kwargs or {}))
        if mixer == "pulay":
            X = orthogonalizer(S)
    S_or_X = S if use_xitorch else GeneralizedDiagonalizer(S).X
    F = T + V_ext
    if P_guess is None:
        P_in, energy_orb = ks_iteration(F, S_or_X, occ, use_xitorch=use_xitorch)
        energy_prev = energy_orb + basis.E_nuc
    else:
        P_in, energy_prev = P_guess, torch.tensor([0e0])
    if enforce_symmetry and isinstance(basis, GridBasis):
        P_in = basis.symmetrize_P(P_in)
    print_iterations = print_iterations if len(P_in.shape) == 2 else False
    if print_iterations:
        print("Iteration | Old energy / Ha | New energy / Ha | Density diff norm")
    for i in iterations or range(max_iterations):
        V_H, V_func, E_func = basis.get_int_integrals(
            P_in, functional, create_graph=create_graph
        )
        F = T + V_ext + V_H + V_func
        if mixer == "pulay":
            err = X.transpose(-1, -2) @ (F @ P_in @ S - S @ P_in @ F) @ X
            F = diis.step(F, err)
        P_out, energy_orb = ks_iteration(F, S_or_X, occ, use_xitorch=use_xitorch)
        # TODO duplicate, should be made part of ks_iteration()
        if enforce_symmetry and isinstance(basis, GridBasis):
            P_out = basis.symmetrize_P(P_out)
        energy = (
            energy_orb
            + E_func
            - ((V_H / 2 + V_func) * P_in).sum((-2, -1))
            + basis.E_nuc
        )
        if tape is not None:
            tape.append((P_out, energy))
        density_diff = basis.density_mse(basis.density(P_out - P_in)).sqrt()
        if print_iterations and i % print_iterations == 0:
            print(
                "%3i   %10.7f   %10.7f   %3.4e" % (i, energy_prev, energy, density_diff)
            )
        converged = density_diff < density_threshold
        if converged.all():
            break
        if mixer == "pulay":
            P_in = P_out
        elif mixer == "pulaydensity":
            P_in = diis.step(P_in, P_out - P_in, alpha)
        elif mixer == "linear":
            alpha_masked = torch.where(converged, 0.0, alpha)[..., None, None]
            P_in = P_in + alpha_masked * (P_out - P_in)
            alpha = alpha * alpha_decay
        energy_prev = energy
    else:
        raise SCFNotConvergedError(P_out, energy)
    return P_out.detach(), energy.detach()
