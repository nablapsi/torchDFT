import sys
import unittest
sys.path.append('../src')

import torch
from functionals import *
from utils import get_dx, coulomb, gaussian

class functionals_test(unittest.TestCase):
    def test_get_hartree_energy(self):
        grid = torch.arange(-5,5,0.1)
        density = gaussian(grid, 1, 1)

        e1 = get_hartree_energy(density, grid, coulomb)

        # Check against nested loop implementation:
        dx = get_dx(grid)
        e2 = 0e0
        for n1, r1 in zip(density, grid):
            for n2, r2 in zip(density, grid):
                disp = r1 - r2
                e2 += n1 * n2 * coulomb(disp)

        e2 *= 5e-1 * dx * dx
        self.assertAlmostEqual(float(e1), float(e2), places=5)

    def test_get_hartree_potential(self):
        grid = torch.arange(-5,5,0.1)
        density = gaussian(grid, 1, 1)

        p1 = get_hartree_potential(density, grid, coulomb)

        # Check against nested loop implementation:
        dx = get_dx(grid)
        p2 = torch.zeros(grid.size(0))
        for i, r in enumerate(grid):
            for n1, r1 in zip(density, grid):
                disp = r1 - r
                p2[i] += n1 * coulomb(disp)

        p2 *= 5e-1 * dx
        self.assertTrue(torch.allclose(p1, p2))


if __name__ == '__main__':
    unittest.main()
