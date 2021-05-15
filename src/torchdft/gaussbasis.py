# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import torch
from pyscf import dft

from .density import Density
from .xc_functionals import LdaPw92


class GaussianBasis:
    """Gaussian basis with radial grids from PySCF."""

    def __init__(self, mol, xc=LdaPw92, kinetic=None):
        self.mol = mol
        self.S = torch.from_numpy(self.mol.intor("int1e_ovlp"))
        self.T = torch.from_numpy(self.mol.intor("int1e_kin"))
        self.V_ext = torch.from_numpy(self.mol.intor("int1e_nuc"))
        self.eri = torch.from_numpy(mol.intor("int2e"))
        self.grid = dft.gen_grid.Grids(mol)
        self.grid.build()
        self.grid_weights = torch.from_numpy(self.grid.weights)
        self.phi = torch.from_numpy(dft.numint.eval_ao(mol, self.grid.coords))
        self.xc = xc()
        self.kinetic = None
        self.E_nuc = mol.energy_nuc()

        if self.kinetic or self.xc.requires_grad:
            raise NotImplementedError(
                "OF-DFT or density gradient not yet implemented for 3D calculations."
            )

    def get_core_integrals(self):
        return self.S, self.T, self.V_ext

    def get_int_integrals(self, P):
        V_H = torch.einsum("ijkl,kl->ij", self.eri, P)
        density = Density(((self.phi @ P) * self.phi).sum(dim=-1))
        density = density.detach()
        density.value = density.value.requires_grad_()
        xc_density = density.value * self.xc.functional(density)
        E_xc = (xc_density.detach() * self.grid_weights).sum()
        (v_xc,) = torch.autograd.grad(xc_density.sum(), density.value)
        V_xc = torch.einsum("g,gi,gj->ij", v_xc * self.grid_weights, self.phi, self.phi)
        return V_H, V_xc, E_xc
