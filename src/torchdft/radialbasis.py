# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Dict, Tuple, Union

import torch
from torch import Tensor

from .basis import Basis
from .density import Density
from .errors import NanError
from .functional import Functional
from .grid import Grid
from .utils import System, SystemBatch, fin_diff_matrix

__all__ = ["RadialBasis"]


class RadialBasis(Basis):
    """Basis of equidistant 1D grid."""

    Z: Tensor

    def __init__(self, system: Union[System, SystemBatch], grid: Grid):
        super().__init__()
        self.system = system
        # In most cases grid is dividing.
        # It is more convenient to change the zeros to infinity.
        self.register_buffer(
            "grid", torch.where(grid.nodes == 0, torch.inf, grid.nodes)
        )
        self.register_buffer("dv", grid.dv)
        self.register_buffer("grid_weights", grid.grid_weights)
        self.register_buffer("Z", self.system.Z.squeeze(-1))
        self.register_buffer("E_nuc", torch.zeros_like(self.Z))
        self.register_buffer("T", -5e-1 * self.get_laplacian())
        self.register_buffer("V_ext", (-self.system.Z / self.grid).diag_embed())
        self.register_buffer(
            "S", self.grid.new_ones(self.grid.size()).diag_embed().unsqueeze(0)
        )
        self.register_buffer(
            "grad_operator",
            fin_diff_matrix(self.grid.shape[-1], 5, 1, dtype=self.grid.dtype)
            / self.grid_weights[..., None, None],
        )

    def get_core_integrals(self) -> Tuple[Tensor, Tensor, Tensor]:
        V_ext_l = self.V_ext
        T = self.T
        S = self.S
        if self.system.lmax > -1:
            V_ext_l = torch.stack((V_ext_l,) * (self.system.lmax + 1), dim=-3)
            for l in range(self.system.lmax + 1):
                V_ext_l[..., l, :, :] += (
                    l * (l + 1) / (2 * self.grid**2)
                ).diag_embed()
            S = S[..., None, :, :]
            T = T[..., None, :, :]
        return S, T, V_ext_l

    def get_int_integrals(
        self,
        P: Tensor,
        functional: Functional,
        create_graph: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Evaluate Hartree and extra potential contributions.

        Args:
            P: Float torch array of dimension (grid_dim, grid_dim)
              holding the density matrix.
            functional: Functional or ComposedFunctional.
              XC, kinetic, or linear combination of functionals.
            create_graph: Bool.
        """
        if not P.requires_grad:
            P = P.detach().requires_grad_()
        grid = torch.where(self.grid.isinf(), self.grid.new_zeros(1), self.grid)
        density = Density(self.density(P), grid, self.dv * self.grid_weights[..., None])
        if functional.requires_grad:
            density.grad = self.get_density_gradient(P)

        V_H = (self.get_hartree_potential(density.value, self.grid)).diag_embed()
        eps_func = functional(density)
        if functional.per_electron:
            eps_func = eps_func * density.density
        E_func = (eps_func * self.dv * self.grid_weights[..., None]).sum(-1)
        (v_func,) = torch.autograd.grad(
            eps_func.sum(), density.value, create_graph=create_graph
        )
        if torch.any(torch.isnan(v_func)):
            raise NanError()
        V_func = v_func.diag_embed()
        if self.system.lmax > -1:
            V_H = V_H[..., None, :, :]
            V_func = V_func[..., None, :, :]
        if not create_graph:
            E_func = E_func.detach()
            V_func = V_func.detach()
            V_H = V_H.detach()
        return V_H, V_func, E_func

    def get_density_gradient(self, P: Tensor) -> Tensor:
        density = self.density(P)
        return torch.einsum(
            "...ij, ...j -> ...i", self.grad_operator, density
        ).squeeze()

    def get_func_laplacian(self, C: Tensor) -> Tensor:
        psi = self.get_psi(C)
        assert psi.shape[0] == self.T.shape[0]
        lap = -2 * torch.einsum("b...ij, b...j -> b...i", self.T, psi)
        return lap

    def density_mse(self, density: Tensor) -> Tensor:
        return (density.pow(2) * self.dv * self.grid_weights[..., None]).sum(dim=-1)

    def density(self, P: Tensor) -> Tensor:
        n = P.diagonal(dim1=-2, dim2=-1) / (self.dv * self.grid_weights[..., None])
        return torch.where(self.dv == 0, n.new_zeros(1), n)

    def quadrupole(self, density: Tensor) -> Tensor:
        grid = torch.where(self.grid.isinf(), self.grid.new_zeros(1), self.grid)
        Q_el = -(grid**2 * density * self.dv * self.grid_weights[..., None]).sum(-1)
        return Q_el

    def density_metrics_fn(
        self, density: Tensor, density_ref: Tensor
    ) -> Dict[str, Tensor]:
        metrics = {}
        Q, Q_ref = (self.quadrupole(x).detach() for x in [density, density_ref])
        metrics["quadrupole"] = ((Q - Q_ref) ** 2).sqrt()
        return metrics

    def get_laplacian(self) -> Tensor:
        """Finite difference approximation of Laplacian operator."""
        grid_dim = self.grid.size(-1)
        laplacian = (
            (-2.5 * self.grid.new_ones([grid_dim]).diag_embed())
            + (4.0 / 3.0 * self.grid.new_ones([grid_dim - 1]).diag_embed(offset=1))
            + (4.0 / 3.0 * self.grid.new_ones([grid_dim - 1]).diag_embed(offset=-1))
            + (-1.0 / 12.0 * self.grid.new_ones([grid_dim - 2]).diag_embed(offset=2))
            + (-1.0 / 12.0 * self.grid.new_ones([grid_dim - 2]).diag_embed(offset=-2))
        )
        if self.Z.size():
            laplacian = torch.stack((laplacian,) * self.Z.shape[0])
        laplacian[..., 0, 0] += (1 + self.grid_weights * self.Z) / (
            12 * (1 - self.grid_weights * self.Z)
        )
        grid_weight_mask = self.dv[..., None, :] * self.dv[..., :, None] == 0.0
        laplacian = torch.where(grid_weight_mask, laplacian.new_zeros(1), laplacian)
        return laplacian / self.grid_weights[..., None, None] ** 2

    def get_hartree_potential(self, density: Tensor, grid: Tensor) -> Tensor:
        """Evaluate Hartree potential."""
        # https://gitlab.com/aheld84/radialdft/-/blob/master/radialdft/poisson.py
        ndv = density * self.dv * self.grid_weights[..., None]
        u_l = grid.new_zeros(density.shape)
        u_l[..., 0] = 0
        u_l[..., 1:] = torch.cumsum(ndv[..., :-1], dim=-1)
        u_l = u_l / grid
        u_r = torch.cumsum((ndv / grid).flip(-1), dim=-1).flip(-1)
        return u_l + u_r

    def get_psi(self, C: Tensor) -> Tensor:
        psi = C / self.grid_weights.sqrt()
        return torch.where(self.grid_weights == 0e0, psi.new_zeros(1), psi)
