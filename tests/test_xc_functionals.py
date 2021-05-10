import torch
from torch.testing import assert_allclose

from torchdft.xc_functionals import (
    get_exponential_coulomb_LDAC_energy_density,
    get_exponential_coulomb_LDAX_energy_density,
)


class TestXcFunctionals:
    def test_get_exponential_coulomb_LDAX_energy_density(self):
        density = torch.Tensor([0.0, 1e-15, 1e-10, 1.0, 5.0])

        p1 = get_exponential_coulomb_LDAX_energy_density(density)
        p2 = torch.Tensor(
            [
                0.0,
                -2.5554081717750003e-15,
                -2.5554081717750006e-10,
                -0.398358052497694,
                -0.49356793671356225,
            ]
        )
        assert_allclose(p1, p2)

    def test_get_exponential_coulomb_LDAC_energy_density(self):
        density = torch.Tensor([0.0, 1e-15, 1e-10, 1.0, 5.0])

        p1 = get_exponential_coulomb_LDAC_energy_density(density)
        p2 = torch.Tensor(
            [
                -0.0,
                -1.2777041412333279e-15,
                -1.2777215850260872e-10,
                -0.00771069759180703,
                -0.0005034570738508356,
            ]
        )
        assert_allclose(p1, p2)
