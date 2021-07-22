import torch
from torch.utils.data import DataLoader

from torchdft.dataset import SystemDataSet, collate_fn
from torchdft.gridbasis import GridBasis
from torchdft.nn_functionals import Conv1dFunctionalNet, GlobalFunctionalNet
from torchdft.train_scf import train_functional
from torchdft.utils import System, exp_coulomb


class TestTrainScf:
    charges = torch.tensor([[1, 0, 1], [1, 1, 1]])
    centers = torch.tensor([[-1, 0, 1], [1, 0, 1]])
    n_electrons = torch.tensor([[2], [3]])
    grid = torch.arange(-10, 10, 1)
    E_truth = torch.rand(charges.shape[0])
    D_truth = torch.rand((charges.shape[0], grid.shape[0]))

    models = [
        Conv1dFunctionalNet(
            window_size=1, channels=[1, 16, 16, 1], negative_transform=True
        ),
        Conv1dFunctionalNet(
            window_size=3, channels=[1, 16, 16, 1], negative_transform=True
        ),
        GlobalFunctionalNet(
            channels=[4, 16, 1],
            glob_channels=4,
            grid=grid,
            kernels=[3, 1],
            maxval=1.0,
            minval=0.0,
            negative_transform=True,
            sig_layer=False,
        ),
        GlobalFunctionalNet(
            channels=[4, 16, 1],
            glob_channels=4,
            grid=grid,
            kernels=[3, 1],
            maxval=1.0,
            minval=0.0,
            negative_transform=True,
            sig_layer=True,
            interaction_fn=exp_coulomb,
        ),
    ]

    systems = []
    for i, charge in enumerate(charges):
        systems.append(
            System(
                charges=charge,
                n_electrons=int(n_electrons[i]),
                centers=centers[i],
                grid=grid,
            )
        )

    systemds = SystemDataSet(systems, E_truth, D_truth)
    dataloader = DataLoader(systemds, batch_size=len(systemds), collate_fn=collate_fn)

    def test_train_scf(self):

        for xc_nn in self.models:
            optimizer = torch.optim.Adam(xc_nn.parameters(), lr=0.01)
            train_functional(
                basis_class=GridBasis,
                functional=xc_nn,
                optimizer=optimizer,
                dataloader=self.dataloader,
                max_epochs=1,
            )

            train_functional(
                basis_class=GridBasis,
                functional=xc_nn,
                optimizer=optimizer,
                dataloader=self.dataloader,
                max_epochs=1,
                enforce_symmetry=True,
            )

    def test_train_scf_closure(self):

        for xc_nn in self.models:
            optimizer = torch.optim.LBFGS(xc_nn.parameters(), lr=1, max_iter=1)
            train_functional(
                basis_class=GridBasis,
                functional=xc_nn,
                optimizer=optimizer,
                dataloader=self.dataloader,
                max_epochs=1,
                requires_closure=True,
            )

            train_functional(
                basis_class=GridBasis,
                functional=xc_nn,
                optimizer=optimizer,
                dataloader=self.dataloader,
                max_epochs=1,
                requires_closure=True,
                enforce_symmetry=True,
            )

    def test_clip_gradients(self):
        xc_nn = self.models[0]
        optimizer = torch.optim.Adam(xc_nn.parameters(), lr=1e2)
        train_functional(
            basis_class=GridBasis,
            functional=xc_nn,
            optimizer=optimizer,
            dataloader=self.dataloader,
            max_epochs=1,
            enforce_symmetry=True,
            max_grad_norm=0.1,
        )

        param = xc_nn.parameters()
        norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in param]))
        assert norm.item() <= 0.1
