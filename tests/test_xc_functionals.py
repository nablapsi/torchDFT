import unittest

import torch

from torchdft.xc_functionals import get_LDAX_potential


class XcFunctionalsTest(unittest.TestCase):
    def test_get_LDAX_potential(self):
        density = torch.Tensor([0.0, 1e-15, 1e-10, 1.0, 5.0])

        p1 = get_LDAX_potential(density)
        p2 = torch.Tensor(
            [
                0.0,
                -5.110816343550001e-15,
                -5.110816343550001e-10,
                -0.4512822906001278,
                -0.5114539643344302,
            ]
        )
        print(p1)
        print(p2)
        self.assertTrue(torch.allclose(p1, p2))
