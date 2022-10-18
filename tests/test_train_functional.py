import shutil

import torch
from torch.testing import assert_allclose

from torchdft.grid import RadialGrid, Uniform1DGrid
from torchdft.gridbasis import GridBasis
from torchdft.nn_functionals import Conv1dFunctionalNet, GlobalFunctionalNet
from torchdft.radialbasis import RadialBasis
from torchdft.scf import RKS
from torchdft.trainingtask import GradientTrainingTask, SCFData, SCFTrainingTask
from torchdft.utils import System, SystemBatch, exp_coulomb
from torchdft.xc_functionals import Lda1d, LdaPw92

torch.set_default_dtype(torch.double)


class TestTrainScf:
    Z = torch.tensor([[1, 0, 1], [1, 1, 1]])
    centers = torch.tensor([[-1, 0, 1], [1, 0, 1]])
    system_list = [System(Z=Zi, centers=ci) for (Zi, ci) in zip(Z, centers)]
    system = SystemBatch(system_list)
    grid = Uniform1DGrid(end=10, dx=0.1, reflection_symmetry=True)
    E_truth = torch.rand(Z.shape[0])
    D_truth = torch.rand((Z.shape[0], grid.nodes.shape[0], grid.nodes.shape[0]))

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
                xc_nn,
                self.basis,
                self.system.occ("OF,RKS"),
                SCFData(self.E_truth, self.D_truth),
                RKS,
                steps=1,
                mixer="linear",
                max_iterations=2,
            )
            task.fit("run/test", device="cpu")
        shutil.rmtree("run")

    def test_train_scf_pulay(self):

        for xc_nn in self.models:
            task = SCFTrainingTask(
                xc_nn,
                self.basis,
                self.system.occ("OF,RKS"),
                SCFData(self.E_truth, self.D_truth),
                RKS,
                steps=1,
                mixer="pulay",
                max_iterations=2,
            )
            task.fit("run/test", device="cpu")
        shutil.rmtree("run")

    def test_train_scf_pulay_adam(self):

        for xc_nn in self.models:
            task = SCFTrainingTask(
                xc_nn,
                self.basis,
                self.system.occ("OF,RKS"),
                SCFData(self.E_truth, self.D_truth),
                RKS,
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
                xc_nn,
                basis,
                self.system_list[0].occ("KS,RKS"),
                SCFData(self.E_truth[0][None], self.D_truth[0][None]),
                RKS,
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
                xc_nn,
                basis,
                self.system_list[0].occ("KS,RKS"),
                SCFData(self.E_truth[0][None], self.D_truth[0][None]),
                RKS,
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


def test_radial_scfmetrics():
    Z = 3
    system = System(centers=torch.tensor([0]), Z=torch.tensor([Z]), spin=Z % 2)
    grid = RadialGrid(end=10, dx=1e-1)
    basis = RadialBasis(system, grid)
    solver = RKS(
        basis,
        system.occ("OF,RKS"),
        LdaPw92(),
    )

    sol = solver.solve(
        mixer="pulaydensity",
        density_threshold=1e-9,
    )

    ref_data = SCFData(sol.E, sol.P)

    task = SCFTrainingTask(
        basis=basis,
        occ=system.occ("OF,RKS"),
        data=ref_data,
        functional=LdaPw92(),
        solver=RKS,
        steps=10,
        mixer="pulaydensity",
        density_threshold=1e-9,
    )
    metrics = task.metrics_fn(task.basis, task.occ, task.data)
    assert_allclose(metrics["loss"], 0e0)


def test_radial_gradmetrics():
    Z = 3
    system = System(centers=torch.tensor([0]), Z=torch.tensor([Z]), spin=Z % 2)
    grid = RadialGrid(end=10, dx=1e-1)
    basis = RadialBasis(system, grid)
    solver = RKS(
        basis,
        system.occ("OF,RKS"),
        LdaPw92(),
    )

    sol = solver.solve(
        mixer="pulaydensity",
        density_threshold=1e-9,
    )

    ref_data = SCFData(sol.E, sol.P, sol.C)

    task = GradientTrainingTask(
        basis=basis,
        occ=system.occ("OF,RKS"),
        data=ref_data,
        functional=LdaPw92(),
        solver=RKS,
        steps=10,
        l=1e0,
    )
    metrics = task.metrics_fn(task.data)
    assert_allclose(metrics["loss"], 0e0)


def test_batchradial_scfmetrics():
    system = SystemBatch(
        [
            System(centers=torch.tensor([0]), Z=torch.tensor([Z]), spin=Z % 2)
            for Z in range(3, 6)
        ]
    )
    grid = RadialGrid(end=10, dx=1e-1)
    basis = RadialBasis(system, grid)
    solver = RKS(
        basis,
        system.occ("OF,RKS"),
        LdaPw92(),
    )

    sol = solver.solve(
        mixer="pulaydensity",
        density_threshold=1e-9,
    )

    ref_data = SCFData(sol.E, sol.P)

    task = SCFTrainingTask(
        basis=basis,
        occ=system.occ("OF,RKS"),
        data=ref_data,
        functional=LdaPw92(),
        solver=RKS,
        steps=10,
        mixer="pulaydensity",
        density_threshold=1e-9,
    )
    metrics = task.metrics_fn(task.basis, task.occ, task.data)
    assert_allclose(metrics["loss"], 0e0)


def test_batchradial_gradmetrics():
    system = SystemBatch(
        [
            System(centers=torch.tensor([0]), Z=torch.tensor([Z]), spin=Z % 2)
            for Z in range(3, 6)
        ]
    )
    grid = RadialGrid(end=10, dx=1e-1)
    basis = RadialBasis(system, grid)
    solver = RKS(
        basis,
        system.occ("OF,RKS"),
        LdaPw92(),
    )

    sol = solver.solve(
        mixer="pulaydensity",
        density_threshold=1e-9,
    )

    ref_data = SCFData(sol.E, sol.P, sol.C)

    task = GradientTrainingTask(
        basis=basis,
        occ=system.occ("OF,RKS"),
        data=ref_data,
        functional=LdaPw92(),
        solver=RKS,
        steps=10,
        l=1e0,
    )
    metrics = task.metrics_fn(task.data)
    assert_allclose(metrics["loss"], 0e0)


def test_grid_scfmetrics():
    system = System(centers=torch.tensor([-1, 0, 1]), Z=torch.tensor([1, 2, 1]))
    grid = Uniform1DGrid(end=10, dx=1e-1, reflection_symmetry=True)
    basis = GridBasis(system, grid)
    solver = RKS(
        basis,
        system.occ("OF,RKS"),
        Lda1d(),
    )

    sol = solver.solve(
        mixer="pulaydensity",
        density_threshold=1e-9,
    )

    ref_data = SCFData(sol.E, sol.P)

    task = SCFTrainingTask(
        basis=basis,
        occ=system.occ("OF,RKS"),
        data=ref_data,
        functional=Lda1d(),
        solver=RKS,
        steps=10,
        mixer="pulaydensity",
        density_threshold=1e-9,
    )
    metrics = task.metrics_fn(task.basis, task.occ, task.data)
    assert_allclose(metrics["loss"], 0e0)


def test_grid_gradmetrics():
    system = System(centers=torch.tensor([-1, 0, 1]), Z=torch.tensor([1, 2, 1]))
    grid = Uniform1DGrid(end=10, dx=1e-1, reflection_symmetry=True)
    basis = GridBasis(system, grid)
    solver = RKS(
        basis,
        system.occ("OF,RKS"),
        Lda1d(),
    )

    sol = solver.solve(
        mixer="pulaydensity",
        density_threshold=1e-9,
    )

    ref_data = SCFData(sol.E, sol.P, sol.C)

    task = GradientTrainingTask(
        basis=basis,
        occ=system.occ("OF,RKS"),
        data=ref_data,
        functional=Lda1d(),
        solver=RKS,
        steps=10,
        l=1e0,
    )
    metrics = task.metrics_fn(task.data)
    assert_allclose(metrics["loss"], 0e0)


def test_batchgrid_scfmetrics():
    system = SystemBatch(
        [
            System(centers=torch.tensor([-1, 0, 1]), Z=torch.tensor([1, 2, 1])),
            System(centers=torch.tensor([-2, 0, 2]), Z=torch.tensor([1, 2, 1])),
        ]
    )
    grid = Uniform1DGrid(end=10, dx=1e-1, reflection_symmetry=True)
    basis = GridBasis(system, grid)
    solver = RKS(
        basis,
        system.occ("OF,RKS"),
        Lda1d(),
    )

    sol = solver.solve(
        mixer="pulaydensity",
        density_threshold=1e-9,
    )

    ref_data = SCFData(sol.E, sol.P, sol.C)

    task = SCFTrainingTask(
        basis=basis,
        occ=system.occ("OF,RKS"),
        data=ref_data,
        functional=Lda1d(),
        solver=RKS,
        steps=10,
        mixer="pulaydensity",
        density_threshold=1e-9,
    )
    metrics = task.metrics_fn(task.basis, task.occ, task.data)
    assert_allclose(metrics["loss"], 0e0)


def test_batchgrid_gradmetrics():
    system = SystemBatch(
        [
            System(centers=torch.tensor([-1, 0, 1]), Z=torch.tensor([1, 2, 1])),
            System(centers=torch.tensor([-2, 0, 2]), Z=torch.tensor([1, 2, 1])),
        ]
    )
    grid = Uniform1DGrid(end=10, dx=1e-1, reflection_symmetry=True)
    basis = GridBasis(system, grid)
    solver = RKS(
        basis,
        system.occ("OF,RKS"),
        Lda1d(),
    )

    sol = solver.solve(
        mixer="pulaydensity",
        density_threshold=1e-9,
    )

    ref_data = SCFData(sol.E, sol.P, sol.C)

    task = GradientTrainingTask(
        basis=basis,
        occ=system.occ("OF,RKS"),
        data=ref_data,
        functional=Lda1d(),
        solver=RKS,
        steps=10,
        l=1e0,
    )
    metrics = task.metrics_fn(task.data)
    assert_allclose(metrics["loss"], 0e0)
