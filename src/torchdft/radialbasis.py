# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import math
from typing import Dict, Tuple, Union

import torch
from torch import Tensor

from .basis import Basis
from .density import Density
from .errors import NanError
from .functional import Functional
from .utils import System, SystemBatch, get_dx

__all__ = ["RadialBasis"]


class RadialBasis(Basis):
    """Basis of equidistant 1D grid."""

    S: Tensor
    T: Tensor
    V_ext: Tensor
    dx: Tensor
    dv: Tensor
    grid: Tensor
    E_nuc: Tensor
    charges: Tensor

    def __init__(
        self,
        system: Union[System, SystemBatch],
    ):
        super().__init__()
        self.system = system
        self.register_buffer("grid", self.system.grid)
        self.register_buffer("dx", get_dx(self.grid))
        self.register_buffer("dv", 4 * math.pi * self.grid ** 2 * self.dx)
        self.register_buffer("charges", self.system.charges)
        self.register_buffer("E_nuc", torch.tensor(0.0))
        self.register_buffer("T", -5e-1 * self.get_laplacian())
        self.register_buffer("V_ext", (-self.system.charges / self.grid).diag_embed())
        self.register_buffer(
            "S",
            torch.full_like(
                self.V_ext[:, 0], 1.0, device=self.grid.device
            ).diag_embed(),
        )

    def get_core_integrals(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.S, self.T, self.V_ext

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
        density = Density(self.density(P))
        density.value = torch.where(
            density.value <= 0.0,
            torch.tensor(
                1e-100, dtype=density.value.dtype, device=density.value.device
            ),
            density.value,
        )
        if functional.requires_grad:
            density.grad = self._get_density_gradient(density.value)

        v_H = self.get_hartree_potential(density.value, self.grid)
        eps_func = functional(density) * density.value
        E_func = (eps_func * self.dv).sum(-1)
        (v_func,) = torch.autograd.grad(
            eps_func.sum(), density.value, create_graph=create_graph
        )
        if not create_graph:
            E_func = E_func.detach()
        if torch.any(torch.isnan(v_func)):
            raise NanError()
        return v_H.diag_embed(), v_func.diag_embed(), E_func

    def _get_density_gradient(self, density: Tensor) -> Tensor:
        grad_operator = self.get_gradient()
        return torch.einsum("ij, ...j -> ...i", grad_operator, density)

    def density_mse(self, density: Tensor) -> Tensor:
        return (density.pow(2) * self.dv).sum(dim=-1)

    def density(self, P: Tensor) -> Tensor:
        return P.diagonal(dim1=-2, dim2=-1) / self.dv

    def quadrupole(self, density: Tensor) -> Tensor:
        Q_el = -(self.grid ** 2 * density * self.dv).sum(-1)
        return Q_el

    def density_metrics_fn(
        self, density: Tensor, density_ref: Tensor
    ) -> Dict[str, Tensor]:
        Q, Q_ref = (self.quadrupole(x).detach() for x in [density, density_ref])
        return {"loss/quadrupole": ((Q - Q_ref) ** 2).mean().sqrt()}

    def get_gradient(self) -> Tensor:
        """Finite difference approximation of gradient operator."""
        grid_dim = self.grid.size(0)
        return (
            (2.0 / 3.0 * self.grid.new_ones([grid_dim - 1]).diag_embed(offset=1))
            + (-2.0 / 3.0 * self.grid.new_ones([grid_dim - 1]).diag_embed(offset=-1))
            + (-1.0 / 12.0 * self.grid.new_ones([grid_dim - 2]).diag_embed(offset=2))
            + (1.0 / 12.0 * self.grid.new_ones([grid_dim - 2]).diag_embed(offset=-2))
        ) / self.dx

    def get_laplacian(self) -> Tensor:
        """Finite difference approximation of Laplacian operator."""
        grid_dim = self.grid.size(0)
        laplacian = (
            (-2.5 * self.grid.new_ones([grid_dim]).diag_embed())
            + (4.0 / 3.0 * self.grid.new_ones([grid_dim - 1]).diag_embed(offset=1))
            + (4.0 / 3.0 * self.grid.new_ones([grid_dim - 1]).diag_embed(offset=-1))
            + (-1.0 / 12.0 * self.grid.new_ones([grid_dim - 2]).diag_embed(offset=2))
            + (-1.0 / 12.0 * self.grid.new_ones([grid_dim - 2]).diag_embed(offset=-2))
        )
        laplacian[0, 0] += (1 + self.dx * self.charges) / (
            12 * (1 - self.dx * self.charges)
        )
        return laplacian / self.dx ** 2

    def get_hartree_potential(self, density: Tensor, grid: Tensor) -> Tensor:
        """Evaluate Hartree potential."""
        # https://gitlab.com/aheld84/radialdft/-/blob/master/radialdft/poisson.py
        ndV = density * self.dv
        u_l = grid.new_zeros(density.shape)
        u_l[0] = 0
        u_l[1:] = torch.cumsum(ndV[:-1], dim=-1)
        u_l = u_l / grid
        u_r = torch.cumsum((ndV / grid).flip(-1), dim=-1).flip(-1)

        return u_l + u_r
