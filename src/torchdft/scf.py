# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import xitorch
import xitorch.linalg
from torch import Tensor

from .basis import Basis
from .errors import SCFNotConvergedError
from .functional import Functional
from .utils import GeneralizedDiagonalizer

DEFAULT_MIXER = "linear"
CONV_TOL = {"E": 0.0, "n": 1e-4}


@dataclass
class SCFSolution:
    """Class holding and SCF solution."""

    E: Tensor
    P: Tensor
    niter: Tensor
    acc_orbital_energy: Tensor
    orbital_energy: Tensor
    C: Tensor
    converged: Tensor


class DensityMixer:
    """Generic class for density mixers."""

    @abstractmethod
    def step(self, P: Tensor, err: Tensor) -> Tensor:
        pass


class SCFSolver(ABC):
    """Generic class for SCF solvers."""

    mixer: DensityMixer

    def __init__(
        self,
        basis: Basis,
        occ: Tensor,
        functional: Functional,
    ):
        self.basis = basis
        self.occ = occ
        self.functional = functional

    def solve(  # noqa: C901 TODO too complex
        self,
        max_iterations: int = 100,
        iterations: Optional[Iterable[int]] = None,
        conv_tol: Optional[Dict[str, float]] = None,
        print_iterations: Union[bool, int] = False,
        tape: Optional[List[Tuple[Tensor, Tensor]]] = None,
        create_graph: bool = False,
        use_xitorch: bool = True,
        mixer: Optional[str] = None,
        mixer_kwargs: Optional[Dict[str, Any]] = None,
        extra_fock_channel: bool = False,
        P_guess: Optional[Tensor] = None,
    ) -> SCFSolution:
        """Given a system, evaluates its energy by solving the KS equations."""
        self.mixer_name = mixer or DEFAULT_MIXER
        self.use_xitorch = use_xitorch
        self.extra_fock_channel = extra_fock_channel
        self.create_graph = create_graph
        self.conv_tol = CONV_TOL.copy()
        if conv_tol is not None:
            self.conv_tol.update(conv_tol)
        assert self.mixer_name in {"linear", "pulay", "pulaydensity"}
        self.S, self.T, self.V_ext = self.basis.get_core_integrals()
        if self.mixer_name in {"pulay", "pulaydensity"}:
            self.mixer = DIIS(**(mixer_kwargs or {}))
        elif self.mixer_name == "linear":
            self.mixer = LinearMixer(**(mixer_kwargs or {}))
        self.S_or_X = self.S if self.use_xitorch else GeneralizedDiagonalizer(self.S).X
        P_in, energy_prev, orbital_energy, C = self.get_init_guess(P_guess)
        P = P_in.sum(-3) if self.extra_fock_channel else P_in
        VH, Vfunc, Efunc = self.basis.get_int_integrals(
            P, self.functional, create_graph=self.create_graph
        )
        energy_prev = self.total_energy(P_in, VH, Efunc)
        for i in iterations or range(max_iterations):
            F = self.build_fock_matrix(P_in, VH, Vfunc)
            P_out, acc_orbital_energy, orbital_energy, C = self.ks_iteration(
                F,
                self.S_or_X,
                self.occ,
            )
            if self.mixer_name in ["pulaydensity" or "linear"]:
                P_out = self.mixer.step(P_in, (P_out - P_in))
            P = P_out.sum(-3) if self.extra_fock_channel else P_out
            VH, Vfunc, Efunc = self.basis.get_int_integrals(
                P, self.functional, create_graph=self.create_graph
            )
            energy = self.total_energy(P_out, VH, Efunc)
            density_diff, converged = self.check_convergence(
                P_in, P_out, energy_prev, energy
            )
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
            P_in = P_out
            energy_prev = energy
        else:
            P_out = P_out.sum(-3) if self.extra_fock_channel else P_out
            raise SCFNotConvergedError(
                SCFSolution(
                    E=energy,
                    P=P_out,
                    niter=torch.tensor(i + 1),
                    acc_orbital_energy=acc_orbital_energy,
                    orbital_energy=orbital_energy,
                    C=C,
                    converged=converged,
                )
            )
        P_out = P_out.sum(-3) if self.extra_fock_channel else P_out
        return SCFSolution(
            E=energy,
            P=P_out,
            niter=torch.tensor(i + 1),
            acc_orbital_energy=acc_orbital_energy,
            orbital_energy=orbital_energy,
            C=C,
            converged=converged,
        )

    def eig(self, F: Tensor, S: Tensor) -> Tuple[Tensor, Tensor]:
        n_occ = self.occ.shape[-1]
        if self.use_xitorch:
            F, S = (xitorch.LinearOperator.m(x, is_hermitian=True) for x in [F, S])
            orbital_energy, C = xitorch.linalg.symeig(F, n_occ, "lowest", S)
        else:
            orbital_energy, C = GeneralizedDiagonalizer.eigh(F, S)
            orbital_energy, C = orbital_energy[..., :n_occ], C[..., :n_occ]
        return orbital_energy, C

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
            acc_orbital_energy: Tensor. Shape = [batch]
            orbital_energy: Tensor. Shape = [batch, (s), (l), orbitals]
            C: Tensor. Shape = [batch, (s), (l), orbitals, basis]

        () => this dimension may not be always present.
        s = spin dimension.
        l = extra Fock matric dimension.
        """
        orbital_energy, C = self.eig(F, S)
        P = (C * occ[..., None, :]) @ C.transpose(
            -2, -1
        )  # shape = [batch, (s), (l), basis, basis]
        acc_orbital_energy = (orbital_energy * occ).sum(
            dim=-1
        )  # shape = [batch, (s), (l)]
        if self.extra_fock_channel:
            acc_orbital_energy = acc_orbital_energy.sum(-1)
        return P, acc_orbital_energy, orbital_energy, C.transpose(-2, -1)

    @abstractmethod
    def get_init_guess(
        self, P_guess: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        pass

    @abstractmethod
    def build_fock_matrix(self, P_in: Tensor, V_H: Tensor, V_func: Tensor) -> Tensor:
        pass

    @abstractmethod
    def check_convergence(
        self, P_in: Tensor, P_out: Tensor, energy_prev: Tensor, energy: Tensor
    ) -> Tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def total_energy(self, P: Tensor, VH: Tensor, Efunc: Tensor) -> Tensor:
        pass


class RKS(SCFSolver):
    """Restricted KS solver."""

    def get_init_guess(
        self, P_guess: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if P_guess is not None:
            if self.extra_fock_channel:
                P_guess = P_guess[:, None, :, :]
            return (
                P_guess,
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
            )
        else:
            return self.ks_iteration(self.T + self.V_ext, self.S_or_X, self.occ)

    def build_fock_matrix(self, P_in: Tensor, V_H: Tensor, V_func: Tensor) -> Tensor:
        F = self.T + self.V_ext + V_H + V_func
        if self.mixer_name == "pulay":
            err = F @ P_in @ self.S - self.S @ P_in @ F
            F = self.mixer.step(F, err)
        return F

    def check_convergence(
        self, P_in: Tensor, P_out: Tensor, energy_prev: Tensor, energy: Tensor
    ) -> Tuple[Tensor, Tensor]:
        P_in = P_in.sum(-3) if self.extra_fock_channel else P_in
        P_out = P_out.sum(-3) if self.extra_fock_channel else P_out
        density_diff = self.basis.density_mse(self.basis.density(P_out - P_in)).sqrt()
        deltaE = (energy_prev - energy).abs()
        return (
            density_diff,
            torch.logical_or(
                density_diff < self.conv_tol["n"], deltaE < self.conv_tol["E"]
            ),
        )

    def total_energy(self, P: Tensor, VH: Tensor, Efunc: Tensor) -> Tensor:
        return (
            ((self.T + self.V_ext + 0.5 * VH) * P).flatten(1).sum(-1)
            + Efunc
            + self.basis.E_nuc
        )


class UKS(SCFSolver):
    """Unrestricted KS solver."""

    def get_init_guess(
        self, P_guess: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if P_guess is not None:
            if self.extra_fock_channel:
                P_guess = P_guess[:, :, None, :, :]
            return (
                P_guess,
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
            )
        else:
            F = self.T + self.V_ext
            F = torch.stack((F, F), 1)
            P, acc_orbital_energy, orbital_energy, C = self.ks_iteration(
                F, self.S_or_X, self.occ
            )
            return (
                P,
                acc_orbital_energy.sum(-1),
                orbital_energy,
                C,
            )

    def build_fock_matrix(self, P_in: Tensor, V_H: Tensor, V_func: Tensor) -> Tensor:
        V_H = V_H[:, None, ...]
        F = self.T[:, None, ...] + self.V_ext[:, None, ...] + V_H + V_func
        if self.mixer_name == "pulay":
            err = F @ P_in @ self.S - self.S @ P_in @ F
            F = self.mixer.step(F, err)
        return F

    def check_convergence(
        self, P_in: Tensor, P_out: Tensor, energy_prev: Tensor, energy: Tensor
    ) -> Tuple[Tensor, Tensor]:
        P_out = P_out.sum(1)
        P_in = P_in.sum(1)
        P_in = P_in.sum(-3) if self.extra_fock_channel else P_in
        P_out = P_out.sum(-3) if self.extra_fock_channel else P_out
        density_diff = self.basis.density_mse(self.basis.density(P_out - P_in)).sqrt()
        deltaE = (energy_prev - energy).abs()
        return (
            density_diff,
            torch.logical_or(
                density_diff < self.conv_tol["n"], deltaE < self.conv_tol["E"]
            ),
        )

    def total_energy(self, P: Tensor, VH: Tensor, Efunc: Tensor) -> Tensor:
        return (
            ((self.T + self.V_ext + 0.5 * VH) * P.sum(1)).flatten(1).sum(-1)
            + Efunc
            + self.basis.E_nuc
        )


class ROKS(UKS):
    """Restricted open KS solver."""

    def build_fock_matrix(self, P_in: Tensor, V_H: Tensor, V_func: Tensor) -> Tensor:
        """
        Build the ROKS Fock matrix.

        Ref: J. Phys. Chem. A, Vol. 114, No. 33, 2010.
        ROHF Fock matrix can be expressed, as a function of alpha and beta
        fock matrices:
                |     closed            open        virtual
        --------|--------------------------------------------------
        closed  | Acc Fa + Bcc Fb        Fb         (Fa + Fb)/2
        open    |        Fb       Aoo Fa + Boo Fb        Fa
        virtual |   (Fa + Fb)/2          Fa       Avv Fa + Bvv Fb

        Arbitrary canonicalization parameters Acc, Aoo, Avv, Bcc, Boo, Bvv
        can be selected. In this implemention we use Davidson canonicalization:
        Acc = Bcc = 1/2, Aoo = Avv = 1, Bcc = Bvv = 0  "in order to reproduce
        UHF occupied and virtual alpha orbital energies when applied to molecule
        with no beta electrons".
        """
        P = P_in.sum(-3) if self.extra_fock_channel else P_in
        V_H, V_func, E_func = self.basis.get_int_integrals(
            P, self.functional, create_graph=self.create_graph
        )
        Fa = self.T + self.V_ext + V_H + V_func[:, 0, ...]
        Fb = self.T + self.V_ext + V_H + V_func[:, 1, ...]
        Pa = P_in[:, 0, ...]
        Pb = P_in[:, 1, ...]

        Fc = (Fa + Fb) * 5e-1
        Pc = torch.einsum("b...ik, b...kj ->b...ij", Pb, self.S)
        Po = torch.einsum("b...ik, b...kj ->b...ij", Pa - Pb, self.S)
        Pv = torch.eye(self.S.shape[-1]) - torch.einsum(
            "b...ik, b...kj ->b...ij", Pa, self.S
        )

        F = (
            5e-1
            * (
                Pc.conj().transpose(-1, -2) @ Fc @ Pc
                + Po.conj().transpose(-1, -2) @ Fa @ Po
                + Pv.conj().transpose(-1, -2) @ Fa @ Pv
            )
            + Po.conj().transpose(-1, -2) @ Fb @ Pc
            + Po.conj().transpose(-1, -2) @ Fa @ Pv
            + Pv.conj().transpose(-1, -2) @ Fc @ Pc
        )
        F = (F + F.conj().transpose(-1, -2)).unsqueeze(1)
        if self.mixer_name == "pulay":
            err = F @ P_in @ self.S - self.S @ P_in @ F
            F = self.mixer.step(F, err)
        return F


class DIIS(DensityMixer):
    """DIIS class."""

    def __init__(
        self,
        max_history: int = 10,
        precondition: bool = False,
        regularization: float = 0e0,
        alpha: Optional[float] = None,
    ) -> None:
        assert not (regularization and not precondition)
        self.max_history = max_history
        self.precondition = precondition
        self.regularization = regularization
        self.history: List[Tuple[Tensor, Tensor]] = []
        self.alpha = alpha

    def _get_coeffs(self, X: Tensor, err: Tensor) -> Tensor:
        self.history.append((X, err))
        self.history = self.history[-self.max_history :]
        nb, N = X.shape[0], len(self.history)
        if N == 1:
            return X.new_ones((nb, 1))
        err = torch.stack([e for _, e in self.history], dim=-2)
        derr = err.diff(dim=-2)
        B = torch.einsum("...im,...jm->...ij", derr, derr)
        y = -torch.einsum("...im,...m->...i", derr, err[..., -1, :])
        if self.precondition:
            pre = 1 / B.detach().diagonal(dim1=-1, dim2=-2).sqrt()
        else:
            pre = B.new_ones((nb, N - 1))
        B = pre[..., None] * pre[..., None, :] * B
        B = B + self.regularization * torch.eye(N - 1, device=B.device)
        c = pre * torch.linalg.solve(B, pre * y)
        c = torch.cat([-c[..., :1], -c.diff(dim=-1), 1 + c[..., -1:]], dim=-1)
        return c

    def step(self, X: Tensor, err: Tensor) -> Tensor:
        err = err.flatten(1)
        c = self._get_coeffs(X, err)
        X = torch.stack([X for X, _ in self.history], dim=1)
        err = torch.stack([e for _, e in self.history], dim=1)
        if self.alpha is not None:
            X = X + self.alpha * err.view(X.shape)
        X = torch.einsum("bi,bi...->b...", c, X)
        return X


class LinearMixer(DensityMixer):
    """Linear density mixer."""

    def __init__(self, alpha: float = 0.5, alpha_decay: float = 1.0):
        self.alpha = alpha
        self.alpha_decay = alpha_decay

    def step(self, P: Tensor, err: Tensor) -> Tensor:
        P_out = P + self.alpha * err
        self.alpha *= self.alpha_decay
        return P_out
