# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Iterable, Tuple, Union

import torch
from pyscf import dft
from pyscf.gto.mole import Mole
from torch import Tensor, from_numpy as fnp

from .basis import Basis
from .density import Density
from .functional import ComposedFunctional, Functional


def _bapply(func, *xs):  # type: ignore
    if isinstance(xs[0], list):
        y = [func(*args) for args in zip(*xs)]
        if isinstance(y[0], Tensor):
            return torch.stack(y)
        return y
    else:
        return func(*xs)


class GaussianBasis(Basis):
    """Gaussian basis with radial grids from PySCF."""

    S: Tensor
    T: Tensor
    V_ext: Tensor
    phi: Tensor

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
        self.grid = _bapply(dft.gen_grid.Grids, self.mol)

        def build_grid(grid):  # type: ignore
            for k, v in kwargs.items():
                setattr(grid, k, v)
            grid.build()

        _bapply(build_grid, self.grid)
        self.register_buffer(
            "grid_weights", _bapply(lambda g: fnp(g.weights), self.grid)
        )
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

    def get_core_integrals(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.S, self.T, self.V_ext

    def density(self, P: Tensor) -> Tensor:
        return ((self.phi @ P) * self.phi).sum(dim=-1)

    def get_int_integrals(
        self,
        P: Tensor,
        xc_functional: Union[Functional, ComposedFunctional],
        create_graph: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        V_H = torch.einsum("...ijkl,...kl->...ij", self.eri, P)
        if not P.requires_grad:
            P = P.detach().requires_grad_()
        density = Density(self.density(P))
        if xc_functional.requires_grad:
            # P + P^t is needed in order for grad w.r.t. P to be symmetric
            density.grad = (
                (
                    (self.phi @ (P + P.transpose(-1, -2)))[..., None, :, :]
                    * self.grad_phi
                )
                .sum(dim=-1)
                .norm(dim=-2)
            )
        E_xc = (density.value * xc_functional(density) * self.grid_weights).sum(dim=-1)
        (V_xc,) = torch.autograd.grad(
            E_xc, P, torch.ones_like(E_xc), create_graph=create_graph
        )
        if not create_graph:
            E_xc = E_xc.detach()
        return V_H, V_xc, E_xc

    def density_mse(self, density: Tensor) -> Tensor:
        return (density ** 2 * self.grid_weights).sum(dim=-1)
