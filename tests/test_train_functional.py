import shutil

import torch

from torchdft.grid import Uniform1DGrid
from torchdft.gridbasis import GridBasis
from torchdft.nn_functionals import Conv1dFunctionalNet, GlobalFunctionalNet
from torchdft.scf import RKS
from torchdft.trainingtask import SCFData, SCFTrainingTask
from torchdft.utils import System, SystemBatch, exp_coulomb


class TestTrainScf:
    Z = torch.tensor([[1, 0, 1], [1, 1, 1]])
    centers = torch.tensor([[-1, 0, 1], [1, 0, 1]])
    system_list = [System(Z=Zi, centers=ci) for (Zi, ci) in zip(Z, centers)]
    system = SystemBatch(system_list)
    grid = Uniform1DGrid(end=10, dx=0.1, reflection_symmetry=True)
    E_truth = torch.rand(Z.shape[0])
    D_truth = torch.rand((Z.shape[0], grid.grid.shape[0], grid.grid.shape[0]))

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

    basis = GridBasis(system, grid)

    def test_train_scf_linear(self):

        for xc_nn in self.models:
            task = SCFTrainingTask(
                RKS,
                xc_nn,
                self.basis,
                self.system.occ("OF,RKS"),
                SCFData(self.E_truth, self.D_truth),
                steps=1,
                mixer="linear",
                max_iterations=2,
            )
            task.fit("run/test", device="cpu")
        shutil.rmtree("run")

    def test_train_scf_pulay(self):

        for xc_nn in self.models:
            task = SCFTrainingTask(
                RKS,
                xc_nn,
                self.basis,
                self.system.occ("OF,RKS"),
                SCFData(self.E_truth, self.D_truth),
                steps=1,
                mixer="pulay",
                max_iterations=2,
            )
            task.fit("run/test", device="cpu")
        shutil.rmtree("run")

    def test_train_scf_pulay_adam(self):

        for xc_nn in self.models:
            task = SCFTrainingTask(
                RKS,
                xc_nn,
                self.basis,
                self.system.occ("OF,RKS"),
                SCFData(self.E_truth, self.D_truth),
                steps=1,
                mixer="pulay",
                max_iterations=2,
            )
            task.fit("run/test", device="cpu", with_adam=True, loss_threshold=0.0)
        shutil.rmtree("run")

    def test_train_scf_pulay_single_basis(self):
        basis = GridBasis(self.system_list[0], self.grid)
        for xc_nn in self.models:
            task = SCFTrainingTask(
                RKS,
                xc_nn,
                basis,
                self.system_list[0].occ("KS,RKS"),
                SCFData(self.E_truth[0][None], self.D_truth[0][None]),
                steps=1,
                mixer="pulay",
                max_iterations=2,
            )
            task.fit("run/test", device="cpu")
        shutil.rmtree("run")

    def test_train_scf_pulay_single_basis_validation(self):
        basis = GridBasis(self.system_list[0], self.grid)
        for xc_nn in self.models:
            task = SCFTrainingTask(
                RKS,
                xc_nn,
                basis,
                self.system_list[0].occ("KS,RKS"),
                SCFData(self.E_truth[0][None], self.D_truth[0][None]),
                steps=1,
                mixer="pulay",
                max_iterations=2,
            )
            task.fit(
                "run/test",
                device="cpu",
                validation_set=(
                    basis,
                    self.system_list[0].occ("KS,RKS"),
                    SCFData(self.E_truth[0][None], self.D_truth[0][None]),
                ),
                validation_step=1,
            )
        shutil.rmtree("run")
