# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from abc import ABC, abstractmethod
from dataclasses import dataclass
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
    F: Tensor,
    S: Tensor,
    occ: Tensor,
    use_xitorch: bool = False,
    extra_fock_channel: bool = False,
) -> Tuple[Tensor, Tensor]:
    n_occ = occ.shape[-1]
    if use_xitorch:
        F, S = (xitorch.LinearOperator.m(x, is_hermitian=True) for x in [F, S])
        epsilon, C = xitorch.linalg.symeig(F, n_occ, "lowest", S)
    else:
        epsilon, C = GeneralizedDiagonalizer.eigh(F, S)
        epsilon, C = epsilon[..., :n_occ], C[..., :n_occ]
    assert epsilon.shape == occ.shape
    if extra_fock_channel:
        C = C.transpose(-3, -2).flatten(start_dim=-2)
        epsilon = epsilon.flatten(start_dim=-2)
        occ = occ.flatten(start_dim=-2)
    P = (C * occ[..., None, :]) @ C.transpose(-2, -1)
    energy_orb = (epsilon * occ).sum(dim=-1)
    return P, energy_orb


@dataclass
class SCFSolution:
    """Class holding and SCF solution."""

    E: Tensor
    P: Tensor
    niter: Tensor
    orbital_energy: Tensor
    epsilon: Tensor
    C: Tensor
    converged: Tensor


class SCFSolver(ABC):
    """Generic class for SCF solvers."""

    def __init__(
        self,
        basis: Basis,
        occ: Tensor,
        functional: Functional,
    ):
        self.basis = basis
        self.occ = occ
        self.functional = functional
        self.asserts()

    def solve(  # noqa: C901 TODO too complex
        self,
        alpha: float = 0.5,
        alpha_decay: float = 1.0,
        max_iterations: int = 100,
        iterations: Iterable[int] = None,
        density_threshold: float = 1e-4,
        print_iterations: Union[bool, int] = False,
        tape: List[Tuple[Tensor, Tensor]] = None,
        create_graph: bool = False,
        use_xitorch: bool = True,
        mixer: str = None,
        mixer_kwargs: Dict[str, Any] = None,
        extra_fock_channel: bool = False,
        P_guess: Tensor = None,
    ) -> SCFSolution:
        """Given a system, evaluates its energy by solving the KS equations."""
        self.mixer = mixer or DEFAULT_MIXER
        self.use_xitorch = use_xitorch
        self.extra_fock_channel = extra_fock_channel
        self.create_graph = create_graph
        assert self.mixer in {"linear", "pulay", "pulaydensity"}
        self.S, self.T, self.V_ext = self.basis.get_core_integrals()
        if self.mixer in {"pulay", "pulaydensity"}:
            self.diis = DIIS(**(mixer_kwargs or {}))
        self.S_or_X = self.S if self.use_xitorch else GeneralizedDiagonalizer(self.S).X
        P_in, energy_prev, epsilon, C = self.get_init_guess(P_guess)
        if tape is not None:
            tape.append((P_in, energy_prev))
        for i in iterations or range(max_iterations):
            F, V_H, V_func, E_func = self.build_fock_matrix(P_in, self.mixer)
            P_out, energy_orb, epsilon, C = self.ks_iteration(
                F,
                self.S_or_X,
                self.occ,
            )
            density_diff, converged = self.check_convergence(
                P_in, P_out, density_threshold
            )
            energy = self.get_total_energy(P_in, V_H, V_func, E_func, energy_orb)
            if tape is not None:
                tape.append((P_out, energy))
            if (
                print_iterations
                and not energy.squeeze().shape
                and i % print_iterations == 0
            ):
                print(
                    f"It: {i:3d}, E = {energy.squeeze():10.7f}, "
                    f"delta_E = {(energy - energy_prev).squeeze():3.4e}, "
                    f"RMSE(n) = {(density_diff).squeeze():3.4e}"
                )
            if converged.all():
                break
            if self.mixer == "pulay":
                P_in = P_out
            elif self.mixer == "pulaydensity":
                P_in = self.diis.step(P_in, (P_out - P_in).flatten(1), alpha)
            elif self.mixer == "linear":
                P_in = P_in + alpha * (P_out - P_in)
                alpha = alpha * alpha_decay
            energy_prev = energy
        else:
            raise SCFNotConvergedError(energy=energy, P=P_out)
        return SCFSolution(
            E=energy,
            P=P_out,
            niter=torch.tensor(i + 1),
            orbital_energy=energy_orb,
            epsilon=epsilon,
            C=C,
            converged=density_diff < density_threshold,
        )

    def eig(self, F: Tensor, S: Tensor) -> Tuple[Tensor, Tensor]:
        n_occ = self.occ.shape[-1]
        if self.use_xitorch:
            F, S = (xitorch.LinearOperator.m(x, is_hermitian=True) for x in [F, S])
            epsilon, C = xitorch.linalg.symeig(F, n_occ, "lowest", S)
        else:
            epsilon, C = GeneralizedDiagonalizer.eigh(F, S)
            epsilon, C = epsilon[..., :n_occ], C[..., :n_occ]
        return epsilon, C

    def ks_iteration(
        self, F: Tensor, S: Tensor, occ: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Diagonalize the Fock matrix to get MO coefficients and energies.

        Input:
            F: Tensor. Shape = [batch, (s), (l), basis, basis].
            S: Tensor. Shape = [batch, (s), (l), basis, basis].
            occ: Tensor. Shape = [batch, (s), (l), orbitals].
        Returns:
            P: Tensor. Shape = [batch, (s), basis, basis]
            energy_orb: Tensor. Shape = [batch]
            epsilon: Tensor. Shape = [batch, (s), (l), orbitals]
            C: Tensor. Shape = [batch, (s), (l), orbitals, basis]

        () => this dimension may not be always present.
        s = spin dimension.
        l = extra Fock matric dimension.
        """
        epsilon, C = self.eig(F, S)
        P = (C * occ[..., None, :]) @ C.transpose(
            -2, -1
        )  # shape = [batch, (s), (l), basis, basis]
        energy_orb = (epsilon * occ).sum(dim=-1)  # shape = [batch, (s), (l)]
        if self.extra_fock_channel:
            P = P.sum(-3)
            energy_orb = energy_orb.sum(-1)
        return P, energy_orb, epsilon, C.transpose(-2, -1)

    def asserts(self) -> None:
        pass

    @abstractmethod
    def get_init_guess(
        self, P_guess: Tensor = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        pass

    @abstractmethod
    def build_fock_matrix(
        self, P_in: Tensor, mixer: str
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        pass

    @abstractmethod
    def get_total_energy(
        self,
        P_in: Tensor,
        V_H: Tensor,
        V_func: Tensor,
        E_func: Tensor,
        energy_orb: Tensor,
    ) -> Tensor:
        pass

    @abstractmethod
    def check_convergence(
        self, P_in: Tensor, P_out: Tensor, density_threshold: float
    ) -> Tuple[Tensor, Tensor]:
        pass


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
        nb, N = X.shape[:-2], len(self.history)
        if N == 1:
            return X.new_ones((*nb, 1))
        err = torch.stack([e for _, e in self.history], dim=-3)
        derr = err.diff(dim=-3)
        B = torch.einsum("...imn,...jmn->...ij", derr, derr)
        y = -torch.einsum("...imn,...mn->...i", derr, err[..., -1, :, :])
        if self.precondition:
            pre = 1 / B.detach().diagonal(dim1=-1, dim2=-2).sqrt()
        else:
            pre = B.new_ones((*nb, N - 1))
        B = pre[..., None] * pre[..., None, :] * B
        B = B + self.regularization * torch.eye(N - 1, device=B.device)
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
    tape: List[Tuple[Tensor, Tensor]] = None,
    create_graph: bool = False,
    use_xitorch: bool = True,
    mixer: str = None,
    P_guess: Tensor = None,
    mixer_kwargs: Dict[str, Any] = None,
    extra_fock_channel: bool = False,
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
        P_in, energy_orb = ks_iteration(
            F,
            S_or_X,
            occ,
            use_xitorch=use_xitorch,
            extra_fock_channel=extra_fock_channel,
        )
        energy_prev = energy_orb + basis.E_nuc
    else:
        P_in, energy_prev = P_guess, torch.tensor([0e0])
    print_iterations = (
        print_iterations if len(P_in.shape) == 2 or P_in.shape[0] == 1 else False
    )
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
        P_out, energy_orb = ks_iteration(
            F,
            S_or_X,
            occ,
            use_xitorch=use_xitorch,
            extra_fock_channel=extra_fock_channel,
        )
        energy = (
            energy_orb
            + E_func
            - ((V_H / 2 + V_func).squeeze() * P_in).sum((-2, -1))
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
