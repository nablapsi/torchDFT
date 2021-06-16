# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Tuple

import torch
from pyscf import dft
from pyscf.gto.mole import Mole
from torch import Tensor

from .basis import Basis
from .density import Density
from .functional import Functional


class GaussianBasis(Basis):
    """Gaussian basis with radial grids from PySCF."""

    def __init__(self, mol: Mole):
        self.mol = mol
        self.S = torch.from_numpy(self.mol.intor("int1e_ovlp"))
        self.T = torch.from_numpy(self.mol.intor("int1e_kin"))
        self.V_ext = torch.from_numpy(self.mol.intor("int1e_nuc"))
        self.eri = torch.from_numpy(mol.intor("int2e"))
        self.grid = dft.gen_grid.Grids(mol)
        self.grid.build()
        self.grid_weights = torch.from_numpy(self.grid.weights)
        phi = torch.from_numpy(dft.numint.eval_ao(mol, self.grid.coords, deriv=1))
        self.phi = phi[0]
        self.grad_phi = phi[1:4]
        self.E_nuc = torch.tensor(mol.energy_nuc())

    def get_core_integrals(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.S, self.T, self.V_ext

    def density(self, P: Tensor) -> Tensor:
        return ((self.phi @ P) * self.phi).sum(dim=-1)

    def get_int_integrals(
        self, P: Tensor, xc_functional: Functional
    ) -> Tuple[Tensor, Tensor, Tensor]:
        V_H = torch.einsum("ijkl,kl->ij", self.eri, P)
        P = P.detach().requires_grad_()
        density = Density(self.density(P))
        if xc_functional.requires_grad:
            # P + P^t is needed in order for grad w.r.t. P to be symmetric
            density.grad = (
                ((self.phi @ (P + P.t())) * self.grad_phi).sum(dim=-1).norm(dim=0)
            )
        E_xc = (density.value * xc_functional(density) * self.grid_weights).sum()
        (V_xc,) = torch.autograd.grad(E_xc, P)
        return V_H, V_xc, E_xc.detach()

    def density_mse(self, density: Tensor) -> Tensor:
        return (density ** 2 * self.grid_weights).sum()
