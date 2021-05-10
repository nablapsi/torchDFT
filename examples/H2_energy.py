import numpy as np
import torch

from torchdft.scf import solve_ks
from torchdft.utils import System

# Specify system:
charges = torch.tensor([1.0, 1.0])
centers = torch.tensor([0.0, 1.401118437])
nelectrons = 2
H2 = System(
    charges=charges,
    centers=centers,
    nelectrons=nelectrons,
    vext=None,
    density=None,
    energy=None,
)

grid = torch.arange(-10, 10, 0.1)

# Solve KS equations
system = solve_ks(H2, grid, print_iterations=True)

density = torch.column_stack((grid, system.density)).numpy()
np.savetxt("H2_density.dat", density)
