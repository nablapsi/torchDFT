# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
from pyscf import dft
from pyscf.gto.mole import Mole
from torch import Tensor, from_numpy as fnp

from .basis import Basis
from .density import Density
from .functional import Functional


def _bapply(func, *xs):  # type: ignore
    if isinstance(xs[0], list):
        y = [func(*args) for args in zip(*xs)]
        if isinstance(y[0], Tensor):
            return torch.stack(y)
        return y
    else:
        return func(*xs)


def _quadrupole(r: Tensor) -> Tensor:
    Q1 = 3 * r[..., None, :] * r[..., :, None]
    Q2 = -r.norm(dim=-1).pow(2)[..., None, None] * torch.eye(3, device=r.device)
    return (Q1 + Q2) / 2


class GaussianBasis(Basis):
    """Gaussian basis with radial grids from PySCF."""

    phi: Tensor
    grid_coords: Tensor
    grid_weights: Tensor
    dv: Tensor
    atom_coords: Tensor
    atom_charges: Tensor
    mask: Optional[Tensor]

    def _intor(self, key: str) -> Tensor:
        return _bapply(lambda m: fnp(m.intor(key)), self.mol)

    def __init__(self, mol: Union[Mole, Iterable[Mole]], **kwargs: object):
        mol = mol if isinstance(mol, Mole) else list(mol)
        super().__init__()
        self.mol = mol
        self.register_buffer("S", self._intor("int1e_ovlp"))
        self.register_buffer("T", self._intor("int1e_kin"))
        self.register_buffer("V_ext", self._intor("int1e_nuc"))
        self.register_buffer("eri", self._intor("int2e"))
        self.register_buffer("dv", self.T.new_ones(1))
        self.grid = _bapply(dft.gen_grid.Grids, self.mol)
        self.mask = None

        def build_grid(grid):  # type: ignore
            for k, v in kwargs.items():
                setattr(grid, k, v)
            grid.build()

        _bapply(build_grid, self.grid)
        self.register_buffer(
            "grid_weights", _bapply(lambda g: fnp(g.weights), self.grid)
        )
        self.register_buffer("grid_coords", _bapply(lambda g: fnp(g.coords), self.grid))
        phi = _bapply(
            lambda m, g: fnp(dft.numint.eval_ao(m, g.coords, deriv=1)),
            self.mol,
            self.grid,
        )
        self.register_buffer("phi", phi[..., 0, :, :])
        self.register_buffer("grad_phi", phi[..., 1:4, :, :])
        self.register_buffer(
            "E_nuc", _bapply(lambda m: torch.tensor(m.energy_nuc()), self.mol)
        )
        self.register_buffer(
            "atom_coords", _bapply(lambda m: fnp(m.atom_coords()), self.mol)
        )
        self.register_buffer(
            "atom_charges", _bapply(lambda m: fnp(m.atom_charges()), self.mol)
        )

    def get_core_integrals(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.S, self.T, self.V_ext

    def density(self, P: Tensor) -> Tensor:
        return ((self.phi @ P) * self.phi).sum(dim=-1)

    def quadrupole(self, density: Tensor) -> Tensor:
        Q_el = (
            (-density * self.grid_weights)[..., None, None]
            * _quadrupole(self.grid_coords)
        ).sum(dim=-3)
        Q_nuc = (
            self.atom_charges[..., None, None] * _quadrupole(self.atom_coords)
        ).sum(dim=-3)
        return Q_el + Q_nuc

    def density_metrics_fn(
        self, density: Tensor, density_ref: Tensor
    ) -> Dict[str, Tensor]:
        metrics = {}
        Q, Q_ref = (
            torch.linalg.eigvalsh(self.quadrupole(x).detach())
            for x in [density, density_ref]
        )
        mse = self.density_mse(density - density_ref)
        metrics["loss/quadrupole"] = ((Q - Q_ref) ** 2).mean(dim=0).sqrt()
        metrics["loss/density_rmse"] = (mse).mean(dim=0).sqrt().detach()
        return metrics

    def get_int_integrals(
        self,
        P: Tensor,
        functional: Functional,
        create_graph: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        V_H = torch.einsum("...ijkl,...kl->...ij", self.eri, P)
        if not P.requires_grad:
            P = P.detach().requires_grad_()
        density = Density(self.density(P), self.grid, self.grid_weights)
        if functional.requires_grad:
            density.grad = self.get_density_gradient(P)
        if self.mask is not None:
            E_func = torch.empty_like(density.value)
            density_tmp = Density(
                density.value[self.mask],
                self.grid,
                self.grid_weights,
                density.grad[self.mask] if density.grad is not None else None,
            )
            E_func[self.mask] = functional(density_tmp)
            for p in functional.parameters():
                p.detach_()
            density_tmp = Density(
                density.value[~self.mask],
                self.grid,
                self.grid_weights,
                density.grad[~self.mask] if density.grad is not None else None,
            )
            E_func[~self.mask] = functional(density_tmp)
            for p in functional.parameters():
                p.requires_grad_()
        else:
            E_func = functional(density)
        if functional.per_electron:
            E_func = density.value * E_func
        E_func = (E_func * self.grid_weights).sum(dim=-1)
        (V_func,) = torch.autograd.grad(
            E_func, P, torch.ones_like(E_func), create_graph=create_graph
        )
        if not create_graph:
            E_func = E_func.detach()
            V_func = V_func.detach()
            V_H = V_H.detach()
        return V_H, V_func, E_func

    def density_mse(self, density: Tensor) -> Tensor:
        return (density**2 * self.grid_weights).sum(dim=-1)

    def get_density_gradient(self, P: Tensor) -> Tensor:
        # P + P^t is needed in order for grad w.r.t. P to be symmetric
        return (
            ((self.phi @ (P + P.transpose(-1, -2)))[..., None, :, :] * self.grad_phi)
            .sum(dim=-1)
            .norm(dim=-2)
        )
