import torch
from torch.testing import assert_allclose

from torchdft.density import Density
from torchdft.gridbasis import GridBasis, get_hartree_potential
from torchdft.nn_functionals import (
    Conv1dFunctionalNet,
    Conv1dPileLayers,
    GgaConv1dFunctionalNet,
    GlobalConvolutionalLayer,
    GlobalFunctionalNet,
    SigLayer,
)
from torchdft.utils import System, exp_coulomb, gaussian


class TestSigLayer:
    grid = torch.arange(-10, 10, 1)
    density = gaussian(grid, 0, 1)

    def test_forward(self):
        layer = SigLayer(self.grid, exp_coulomb)
        xc_energy_density = torch.rand(self.grid.shape)

        _ = layer(self.density, xc_energy_density)

    def test_1e(self):
        layer = SigLayer(self.grid, exp_coulomb)
        xc_energy_density = torch.rand(self.grid.shape)

        out = layer(self.density, xc_energy_density)
        V_H = get_hartree_potential(self.density, self.grid, exp_coulomb)

        assert_allclose(out.squeeze(), -5e-1 * V_H)


class TestGlobalConvolutionalLayer:
    def test_forward(self):
        nbatch, ngrid, nchannels = 2, 3, 4
        grid = torch.rand(ngrid)
        density = torch.rand(nbatch, 1, ngrid)

        gconvlayer = GlobalConvolutionalLayer(nchannels, grid)

        value = gconvlayer(density)

        xi = 1 / (
            gconvlayer.minval
            + (gconvlayer.maxval - gconvlayer.minval) * torch.sigmoid(gconvlayer.xi)
        )
        # Unrolled calculation.
        a = torch.zeros([nchannels, ngrid, ngrid])
        for i in range(nchannels):
            for j in range(ngrid):
                for k in range(ngrid):
                    a[i, j, k] += gconvlayer.g[j, k] * xi[i]

        a = (-a).exp()

        uvalue = torch.zeros(nbatch, nchannels, ngrid)
        for i in range(nbatch):
            for j in range(nchannels):
                for k in range(ngrid):
                    for l in range(ngrid):
                        uvalue[i, j, l] += density[i, 0, k] * a[j, k, l] * xi[j]
        uvalue *= 5e-1 * gconvlayer.dx

        assert_allclose(value, uvalue)


class TestConv1dPileLayers:
    channels = [1, 16, 16, 1]
    kernels = [1, 3, 1]

    def test_forward(self):
        layer = Conv1dPileLayers(self.channels, self.kernels, negative_transform=True)

        density = torch.rand((1, 1, 10))
        out = layer(density)

        assert list(out.shape) == [10]

    def test_forward_batch(self):
        layer = Conv1dPileLayers(self.channels, self.kernels, negative_transform=True)

        density = torch.rand((3, 1, 10))
        out = layer(density)

        assert list(out.shape) == [3, 10]


class TestConv1dFunctionalNet:
    def test_LDA_forward(self):
        net = Conv1dFunctionalNet(1, [1, 16, 16, 1], negative_transform=True)
        density = Density(torch.rand((3, 10)))

        _ = net(density)

    def test_GGA_forward(self):
        net = Conv1dFunctionalNet(3, [1, 16, 16, 1], negative_transform=True)
        density = Density(torch.rand((3, 10)))

        _ = net(density)


class TestGlobalFunctionalNet:
    grid = torch.arange(-10, 10, 1)
    density = Density(gaussian(grid, 0, 1))

    def test_forward(self):
        net = GlobalFunctionalNet(
            channels=[4, 16, 1],
            glob_channels=4,
            grid=self.grid,
            kernels=[3, 1],
            maxval=1.0,
            minval=0.0,
            negative_transform=True,
            sig_layer=False,
        )

        _ = net(self.density)

    def test_SIG_forward(self):
        net = GlobalFunctionalNet(
            channels=[4, 16, 1],
            glob_channels=4,
            grid=self.grid,
            kernels=[3, 1],
            maxval=1.0,
            minval=0.0,
            negative_transform=True,
            sig_layer=True,
            interaction_fn=exp_coulomb,
        )

        _ = net(self.density)


class TestGgaConv1dFunctionalNet:
    grid = torch.arange(-10, 10, 1)
    system = System(
        centers=torch.tensor([0.0]),
        charges=torch.tensor([1]),
        n_electrons=1,
        grid=grid,
    )

    basis = GridBasis(system)

    def test_forward(self):
        net = GgaConv1dFunctionalNet(channels=[2, 16, 16, 1], negative_transform=True)
        density = Density(torch.rand((3, self.grid.shape[0])))
        density.grad = self.basis._get_density_gradient(density.value)

        _ = net(density)
