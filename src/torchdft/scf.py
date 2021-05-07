# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import torch

from torchdft.functionals import (
    get_external_potential,
    get_external_potential_energy,
    get_hartree_energy,
    get_hartree_potential,
    get_kinetic_matrix,
    get_XC_energy,
    get_XC_potential,
)
from torchdft.utils import exp_coulomb, get_dx
from torchdft.xc_functionals import exponential_coulomb_LDA_XC_energy_density


def get_effective_potential(ext_pot, density, grid, interaction_fn, XC_energy_density):
    """Evaluate the effective potential of the system."""
    return (
        ext_pot
        + get_hartree_potential(density, grid, interaction_fn)
        + get_XC_potential(density, grid, XC_energy_density)
    )


def get_hamiltonian_matrix(grid, veff):
    """Evaluate the hamiltonian matrix."""
    return get_kinetic_matrix(grid) + torch.diag_embed(veff)


def get_density_from_wf(nelectrons, grid, eigstates):
    """Evaluate the density from a set of orbitals."""
    dx = get_dx(grid)

    # Only nelectrons / 2 spatial orbitals are occupied:
    nocc = nelectrons // 2
    occ = eigstates[:, :nocc]

    occ = occ / torch.sqrt(torch.sum(occ * occ, axis=0) * dx)

    occ = torch.column_stack((occ ** 2,) * 2)
    return torch.sum(occ, axis=1)


def get_total_eigen_ener(nelectrons, eigener):
    """Evaluate the sum of eigen energies for the occupied states."""
    nocc = nelectrons // 2
    occ = eigener[:nocc]
    occ = torch.column_stack((occ,) * 2)
    return torch.sum(occ)


def get_total_energy(
    system, eigen_ener, veff, grid, density, interaction_fn, XC_energy_density
):
    """Evaluate the total energy of the system."""
    E_veff = get_external_potential_energy(veff, density, grid)
    K_ener = eigen_ener - E_veff

    return (
        K_ener
        + get_external_potential_energy(system.vext, density, grid)
        + get_hartree_energy(density, grid, interaction_fn)
        + get_XC_energy(density, grid, XC_energy_density)
    )


def ks_iteration(nelectrons, veff, grid):
    """Run one iteration to solve the KS equations."""
    # Generate Hamiltonian matrix.
    H = get_hamiltonian_matrix(grid, veff)

    # Solve eigenstates:
    eigener, eigstates = torch.linalg.eigh(H)

    # Generate new density:
    new_density = get_density_from_wf(nelectrons, grid, eigstates)

    # Total eigen energy:
    total_eigener = get_total_eigen_ener(nelectrons, eigener)

    return new_density, total_eigener


def solve_ks(
    system,
    grid,
    alpha=0.5,
    interaction_fn=exp_coulomb,
    XC_energy_density=exponential_coulomb_LDA_XC_energy_density,
    max_iterations=100,
    print_iterations=False,
):
    """Given a system, evaluates its energy solvig the KS equations."""

    # Get external potential.
    system.vext = get_external_potential(
        system.charges, system.centers, grid, interaction_fn
    )

    # First iteration will use vext as veff since we don't have any density.
    new_density, total_eigener = ks_iteration(system.nelectrons, system.vext, grid)
    new_ener = 0e0

    if print_iterations:
        print("Iteration | Old energy / Ha | New energy / Ha | Absolute difference")
    for it in range(1, max_iterations + 1):
        old_density = new_density.clone().detach()
        old_ener = new_ener

        veff = get_effective_potential(
            system.vext, old_density, grid, interaction_fn, XC_energy_density
        )
        density, total_eigener = ks_iteration(system.nelectrons, veff, grid)
        new_ener = get_total_energy(
            system,
            total_eigener,
            veff,
            grid,
            density,
            interaction_fn,
            XC_energy_density,
        )

        old_density.requires_grad = False
        new_density = old_density + alpha * (density - old_density)

        if print_iterations:
            print(
                "%3i   %10.7f   %10.7f   %3.4e"
                % (it, old_ener, new_ener, torch.abs(old_ener - new_ener))
            )
            # it, old_ener, total_ener)

        # TODO: Add some kind of convergence criteria so not all iterations are evaluated.
        # if torch.abs(old_ener - total_ener) < 1e-5:
        #    break

    return new_density, new_ener
