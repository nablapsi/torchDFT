import shutil

import torch

from torchdft.gridbasis import GridBasis
from torchdft.nn_functionals import Conv1dFunctionalNet, GlobalFunctionalNet
from torchdft.train_scf import SCFData, TrainingTask
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
            window_size=1, channels=[1, 16, 1], negative_transform=True
        ),
        Conv1dFunctionalNet(
            window_size=3, channels=[1, 16, 1], negative_transform=True
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

    basislist = []
    for i, charge in enumerate(charges):
        basislist.append(
            GridBasis(
                System(
                    charges=charge,
                    n_electrons=int(n_electrons[i]),
                    centers=centers[i],
                    grid=grid,
                )
            )
        )

    def test_train_scf_linear(self):

        for xc_nn in self.models:
            task = TrainingTask(
                xc_nn,
                self.basislist,
                self.n_electrons,
                SCFData(self.E_truth, self.D_truth),
                steps=1,
                mixer="linear",
                max_iterations=2,
                enforce_symmetry=True,
            )
            task.fit("run/test", device="cpu")
        shutil.rmtree("run")

    def test_train_scf_pulay(self):

        for xc_nn in self.models:
            task = TrainingTask(
                xc_nn,
                self.basislist,
                self.n_electrons,
                SCFData(self.E_truth, self.D_truth),
                steps=1,
                mixer="pulay",
                max_iterations=2,
                enforce_symmetry=True,
            )
            task.fit("run/test", device="cpu")
        shutil.rmtree("run")
