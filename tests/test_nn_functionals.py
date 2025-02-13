import torch
from torch.testing import assert_allclose

from torchdft.density import Density
from torchdft.grid import Uniform1DGrid
from torchdft.gridbasis import GridBasis, get_hartree_potential
from torchdft.nn_functionals import (
    Conv1dFunctionalNet,
    Conv1dPileLayers,
    GgaConv1dFunctionalNet,
    GlobalConvolutionalLayer,
    GlobalFunctionalNet,
    LDANet,
    NDVConvNet,
    NDVNConvLogNet,
    NDVNConvLogPolarizedNet,
    NDVNConvNet,
    NDVNConvNetAlpha,
    NDVNConvNetAlphaGrid,
    NDVNConvNetGrid,
    NDVNConvNetLogGrid,
    NDVNet,
    PolarizedLDANet,
    SigLayer,
    TrainableGGA,
    TrainableLDA,
)
from torchdft.utils import System, exp_coulomb, gaussian


class TestSigLayer:
    grid = Uniform1DGrid(end=10, dx=1e0, reflection_symmetry=True)
    density = gaussian(grid.nodes, 0, 1)[None, :]

    def test_forward(self):
        layer = SigLayer(self.grid, exp_coulomb)
        xc_energy_density = torch.rand(self.grid.nodes.shape)

        _ = layer(self.density[None, ...], xc_energy_density)

    def test_1e(self):
        layer = SigLayer(self.grid, exp_coulomb)
        xc_energy_density = torch.rand(self.grid.nodes.shape)

        out = layer(self.density[None, ...], xc_energy_density)
        V_H = get_hartree_potential(self.density, self.grid.nodes, exp_coulomb)

        assert_allclose(out, -5e-1 * V_H)


class TestGlobalConvolutionalLayer:
    def test_forward(self):
        grid = Uniform1DGrid(end=10, dx=1e0, reflection_symmetry=True)
        nbatch, ngrid, nchannels = 2, grid.nodes.shape[0], 4
        density = torch.rand(nbatch, 1, ngrid)

        gconvlayer = GlobalConvolutionalLayer(nchannels, grid)

        value = gconvlayer(density)

        xi = 1 / (
            gconvlayer.minval
            + (gconvlayer.maxval - gconvlayer.minval) * torch.sigmoid(gconvlayer.xi)
        )
        # Unrolled calculation.
        a = torch.zeros([nchannels - 1, ngrid, ngrid])
        for i in range(nchannels - 1):
            for j in range(ngrid):
                for k in range(ngrid):
                    a[i, j, k] += gconvlayer.g[j, k] * xi[i]

        a = (-a).exp()

        uvalue = torch.zeros(nbatch, nchannels - 1, ngrid)
        for i in range(nbatch):
            for j in range(nchannels - 1):
                for k in range(ngrid):
                    for l in range(ngrid):
                        uvalue[i, j, l] += density[i, 0, k] * a[j, k, l] * xi[j]
        uvalue *= 5e-1 * gconvlayer.grid_weights
        uvalue = torch.cat((uvalue, density), dim=1)

        assert_allclose(value, uvalue)


class TestConv1dPileLayers:
    channels = [1, 16, 16, 1]
    kernels = [1, 3, 1]

    def test_forward(self):
        layer = Conv1dPileLayers(self.channels, self.kernels, negative_transform=True)

        density = torch.rand((1, 1, 10))
        out = layer(density)

        assert list(out.shape) == [1, 10]

    def test_forward_batch(self):
        layer = Conv1dPileLayers(self.channels, self.kernels, negative_transform=True)

        density = torch.rand((3, 1, 10))
        out = layer(density)

        assert list(out.shape) == [3, 10]


class TestConv1dFunctionalNet:
    def test_LDA_forward(self):
        net = Conv1dFunctionalNet(1, [1, 16, 16, 1], negative_transform=True)
        density = Density(torch.rand((3, 10)), torch.rand(10), torch.rand(10))

        _ = net(density)

    def test_GGA_forward(self):
        net = Conv1dFunctionalNet(3, [1, 16, 16, 1], negative_transform=True)
        density = Density(torch.rand((3, 10)), torch.rand(10), torch.rand(10))

        _ = net(density)


class TestGlobalFunctionalNet:
    grid = Uniform1DGrid(end=10, dx=1e0, reflection_symmetry=True)
    density = Density(gaussian(grid.nodes, 0, 1), grid.nodes, grid.grid_weights)

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
    grid = Uniform1DGrid(end=10, dx=0.1, reflection_symmetry=True)
    system = System(
        centers=torch.tensor([0.0]),
        Z=torch.tensor([1]),
    )

    basis = GridBasis(system, grid)

    def test_forward(self):
        net = GgaConv1dFunctionalNet(channels=[2, 16, 16, 1], negative_transform=True)
        density = Density(
            torch.rand((3, self.grid.nodes.shape[0])),
            self.grid.nodes,
            self.grid.grid_weights,
        )
        density.grad = self.basis.get_density_gradient(density.value.diag_embed())
        _ = net(density)


class TestRadialGlobalFunctionalNet:
    density = Density(torch.rand((3, 10)), torch.rand(10), torch.rand(10))

    def test_NDVNet(self):
        net = NDVNet()
        _ = net(self.density)

    def test_NDVConvNet(self):
        net = NDVConvNet()
        _ = net(self.density)

    def test_NDVNConvNet(self):
        net = NDVNConvNet()
        _ = net(self.density)

    def test_NDVNConvLogNet(self):
        net = NDVNConvLogNet()
        _ = net(self.density)

    def test_NDVNConvNetAlpha(self):
        net = NDVNConvNetAlpha()
        _ = net(self.density)


class TestPolarizedNets:
    density = Density(torch.rand((3, 10)), torch.rand(10), torch.rand(10))

    def test_NDVNConvLogPolarizedNet(self):
        net = NDVNConvLogPolarizedNet(layers=[2, 2, 2])
        _ = net(self.density)

    def test_PolarizedLDANet(self):
        net = PolarizedLDANet()
        _ = net(self.density)


class TestTrainableLDA:
    net = TrainableLDA()
    density = Density(torch.rand((3, 10)), torch.rand(10), torch.rand(10))
    _ = net(density)


class TestTrainableGGA:
    net = TrainableGGA()
    density = Density(torch.rand((3, 10)), torch.rand(10), torch.rand(10))
    density.grad = torch.rand((3, 10))
    _ = net(density)


class TestGridFunctionals:
    density = Density(torch.rand((3, 10)), torch.rand(10), torch.rand(10))

    def test_NDVNConvNet(self):
        net = NDVNConvNetGrid()
        _ = net(self.density)

    def test_NDVNConvNetAlphaGrid(self):
        net = NDVNConvNetAlphaGrid()
        _ = net(self.density)

    def test_NDVNConvNetLogGrid(self):
        net = NDVNConvNetLogGrid()
        _ = net(self.density)


class TestLDANet:
    net = LDANet()
    density = Density(torch.rand((3, 10)), torch.rand(10), torch.rand(10))
    _ = net(density)
