import shutil

import torch

from torchdft.grid import Uniform1DGrid
from torchdft.gridbasis import GridBasis
from torchdft.nn_functionals import Conv1dFunctionalNet, GlobalFunctionalNet
from torchdft.trainingtask import SCFData, TrainingTask
from torchdft.utils import System, exp_coulomb


class TestTrainScf:
    Z = torch.tensor([[1, 0, 1], [1, 1, 1]])
    centers = torch.tensor([[-1, 0, 1], [1, 0, 1]])
    n_electrons = torch.tensor([[2], [3]])
    grid = Uniform1DGrid(end=10, dx=0.1, reflection_symmetry=True)
    E_truth = torch.rand(Z.shape[0])
    D_truth = torch.rand((Z.shape[0], grid.grid.shape[0]))

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
            grid=grid.grid,
            kernels=[3, 1],
            maxval=1.0,
            minval=0.0,
            negative_transform=True,
            sig_layer=False,
        ),
        GlobalFunctionalNet(
            channels=[4, 16, 1],
            glob_channels=4,
            grid=grid.grid,
            kernels=[3, 1],
            maxval=1.0,
            minval=0.0,
            negative_transform=True,
            sig_layer=True,
            interaction_fn=exp_coulomb,
        ),
    ]

    basislist = []
    for i, Zi in enumerate(Z):
        basislist.append(
            GridBasis(
                System(
                    Z=Zi,
                    centers=centers[i],
                ),
                grid,
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
            )
            task.fit("run/test", device="cpu")
        shutil.rmtree("run")

    def test_train_scf_pulay_adam(self):

        for xc_nn in self.models:
            task = TrainingTask(
                xc_nn,
                self.basislist,
                self.n_electrons,
                SCFData(self.E_truth, self.D_truth),
                steps=1,
                mixer="pulay",
                max_iterations=2,
            )
            task.fit("run/test", device="cpu", with_adam=True, loss_threshold=0.0)
        shutil.rmtree("run")

    def test_train_scf_pulay_single_basis(self):
        system = System(
            Z=self.Z[0],
            centers=self.centers[0],
        )
        basis = GridBasis(system, self.grid)
        for xc_nn in self.models:
            task = TrainingTask(
                xc_nn,
                basis,
                system.occ("KS"),
                SCFData(self.E_truth[0], self.D_truth[0]),
                steps=1,
                mixer="pulay",
                max_iterations=2,
            )
            task.fit("run/test", device="cpu")
        shutil.rmtree("run")

    def test_train_scf_pulay_single_basis_validation(self):
        system = System(
            Z=self.Z[0],
            centers=self.centers[0],
        )
        basis = GridBasis(system, self.grid)
        for xc_nn in self.models:
            task = TrainingTask(
                xc_nn,
                basis,
                system.occ("KS"),
                SCFData(self.E_truth[0], self.D_truth[0]),
                steps=1,
                mixer="pulay",
                max_iterations=2,
            )
            task.fit(
                "run/test",
                device="cpu",
                validation_set=(
                    basis,
                    system.occ("KS"),
                    SCFData(self.E_truth[0], self.D_truth[0]),
                ),
                validation_step=1,
            )
        shutil.rmtree("run")
