# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import annotations

import logging
import random
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .basis import Basis
from .density import Density
from .errors import GradientError, SCFNotConvergedError
from .functional import Functional
from .gaussbasis import GaussianBasis
from .scf import SCFSolver

__all__ = ["TrainingTask"]
SCHEDULER_KWARGS = {"patience": 300, "min_lr": 1e-6}
Metrics = Dict[str, Tensor]
LossFn = Callable[
    [Basis, Tensor, Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Metrics]
]

log = logging.getLogger(__name__)


class SCFData(nn.Module):
    energy: Tensor
    P: Tensor
    C: Optional[Tensor]

    def __init__(self, energy: Tensor, P: Tensor, C: Optional[Tensor] = None):
        super().__init__()
        self.register_buffer("energy", energy)
        self.register_buffer("P", P)
        self.register_buffer("C", C)


class GradTTData(nn.Module):
    N: Tensor
    occ_mask: Tensor
    psi: Tensor
    grid: Tensor
    grid_weights: Tensor
    dv: Tensor
    n: Tensor
    TVextVH: Tensor
    energybase: Tensor
    Eref: Tensor

    def __init__(
        self,
        N: Tensor,
        occ_mask: Tensor,
        psi: Tensor,
        grid: Tensor,
        grid_weights: Tensor,
        dv: Tensor,
        n: Tensor,
        TVextVH: Tensor,
        energybase: Tensor,
        Eref: Tensor,
    ):
        super().__init__()
        self.gridbatch = len(grid.shape) == 2
        self.register_buffer("N", N)
        self.register_buffer("occ_mask", occ_mask)
        self.register_buffer("psi", psi)
        self.register_buffer("grid", grid)
        self.register_buffer("grid_weights", grid_weights)
        self.register_buffer("dv", dv)
        self.register_buffer("n", n)
        self.register_buffer("TVextVH", TVextVH)
        self.register_buffer("energybase", energybase)
        self.register_buffer("Eref", Eref)

    def __getitem__(self, item: List[int]) -> GradTTData:
        return GradTTData(
            self.N[item],
            self.occ_mask[item],
            self.psi[item],
            self.grid,
            self.grid_weights,
            self.dv,
            self.n[item],
            self.TVextVH[item],
            self.energybase[item],
            self.Eref[item],
        )

    def __len__(self) -> int:
        return self.N.shape[0]


class GradTTDataBasis(nn.Module):
    N: Tensor
    occ_mask: Tensor
    C: Tensor
    S: Tensor
    TVext: Tensor
    energybase: Tensor
    Eref: Tensor

    def __init__(
        self,
        N: Tensor,
        occ_mask: Tensor,
        C: Tensor,
        S: Tensor,
        TVext: Tensor,
        energybase: Tensor,
        Eref: Tensor,
    ):
        super().__init__()
        self.register_buffer("N", N)
        self.register_buffer("occ_mask", occ_mask)
        self.register_buffer("C", C)
        self.register_buffer("S", S)
        self.register_buffer("TVext", TVext)
        self.register_buffer("energybase", energybase)
        self.register_buffer("Eref", Eref)

    def __getitem__(self, item: List[int]) -> GradTTDataBasis:
        if self.TVext.shape[0] > 1:
            TVext = self.TVext[item]
        return GradTTDataBasis(
            self.N[item],
            self.occ_mask[item],
            self.C[item],
            self.S,
            TVext,
            self.energybase[item],
            self.Eref[item],
        )

    def __len__(self) -> int:
        return self.N.shape[0]


class DataLoader:
    def __init__(
        self,
        data: Union[GradTTData, GradTTDataBasis],
        batchsize: int,
        shuffle: bool = True,
    ):
        self.data = data
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.datasize = len(self.data)

        self.data_indexes = list(range(self.datasize))
        self.nminimibatch = self.datasize // self.batchsize + (
            int(bool(self.datasize % self.batchsize))
        )

    def __iter__(self) -> DataLoader:
        self.n = 0
        if self.shuffle:
            random.shuffle(self.data_indexes)
        return self

    def __next__(self) -> Union[GradTTData, GradTTDataBasis]:
        if self.n < self.nminimibatch:
            batch = self.data_indexes[
                self.n * self.batchsize : (self.n + 1) * self.batchsize
            ]
            self.n += 1
            return self.data[batch]
        else:
            raise StopIteration


class TqdmStream:
    def write(self, msg: str) -> int:
        try:
            tqdm.write(msg, end="")
        except BrokenPipeError:
            sys.stderr.write(msg)
            return 0
        return len(msg)


class CheckpointStore:
    def __init__(self) -> None:
        self.chkpts: List[Path] = []

    def replace(self, state: Dict[str, Any], path: Path) -> None:
        self.chkpts.append(path)
        while len(self.chkpts) > 1:
            self.chkpts.pop(0).unlink()
        torch.save(state, self.chkpts[-1])


class TrainingTask(nn.Module, ABC):
    """Represents a training task."""

    make_solver: Optional[type[SCFSolver]]
    basis: Basis
    occ: Tensor
    data: Union[SCFData, GradTTData, GradTTDataBasis]
    train_samples: int
    functional: Functional
    steps: int
    supports_minibatch = False

    def prepare_validation_data(
        self,
        basis: Basis,
        occ: Tensor,
        data: Union[SCFData, Tuple[Union[float, Tensor], Tensor]],
    ) -> Tuple[Basis, Tensor, SCFData, int]:
        if not isinstance(data, SCFData):
            energy, P = data
            energy = torch.as_tensor(energy)
            data = SCFData(energy, P)
        if not basis.E_nuc.shape:  # single basis
            samples = 1
            assert len(occ.shape) == 1
            assert len(data.energy.shape) == 0
            assert len(data.P.shape) == 2
        else:
            samples = basis.E_nuc.shape[0]
            assert len(basis.E_nuc.shape) == 1
            assert len(data.energy.shape) == 1
            assert 3 <= len(data.P.shape) <= 4
            assert occ.shape[0] == samples
            assert data.energy.shape[0] == samples
            assert data.P.shape[0] == samples
        return basis, occ, data, samples

    @abstractmethod
    def metrics_fn(self, data: Union[SCFData, GradTTData, GradTTDataBasis]) -> Metrics:
        pass

    def training_step(
        self,
        data: Union[SCFData, GradTTData, GradTTDataBasis],
        opt: Optional[torch.optim.Optimizer] = None,
    ) -> Metrics:
        """Execute a training step."""
        assert self.training
        metrics = self.metrics_fn(data)
        # Evaluate (d RMSE / d theta) from (d MSE / d theta)
        for p in self.functional.parameters():
            p.grad = p.grad / (2.0 * metrics["loss"])
        if opt is not None:
            opt.step()
            opt.zero_grad()
        assert not any(v.grad_fn for v in metrics.values())
        return metrics

    def validation_eval(
        self, basis: Basis, occ: Tensor, data: SCFData, **kwargs: Any
    ) -> Tuple[Tensor, Metrics]:
        assert self.make_solver is not None
        solver = self.make_solver(basis, occ, self.functional)
        try:
            sol = solver.solve(P_guess=data.P, **kwargs)
        except SCFNotConvergedError as e:
            sol = e.sol
        npred = basis.density(sol.P)
        nref = basis.density(data.P)
        if len(sol.P.shape) == 4:
            npred = npred.sum(1)
            nref = nref.sum(1)
        Eloss = (sol.E - data.energy).abs()
        nloss = basis.density_mse(npred - nref)
        nmetrics = basis.density_metrics_fn(npred, nref)
        Eloss[sol.converged.logical_not()] = torch.nan
        nloss[sol.converged.logical_not()] = torch.nan
        metrics = {}
        for key in nmetrics.keys():
            nmetrics[key][sol.converged.logical_not()] = torch.nan
        if (
            data.P.shape[0] > 1
            and self.minibatchsize is None
            and self.store_individual_loss
        ):
            for i, (ener, den) in enumerate(zip(Eloss, nloss)):
                metrics[f"individual_validation/Eloss{i}"] = ener.detach()
                metrics[f"individual_validation/nloss{i}"] = den.detach().sqrt()
                for key in nmetrics.keys():
                    metrics[f"individual_validation/{key}{i}"] = nmetrics[key][i]
        Eloss, nloss = Eloss.mean(), nloss.mean()
        loss = Eloss**2 + nloss
        metrics["validation/SCFiter"] = sol.niter
        metrics["validation/Eloss"] = Eloss
        metrics["validation/nloss"] = nloss.sqrt()
        metrics["validation/loss"] = loss.sqrt()
        for key in nmetrics.keys():
            metrics[f"validation/{key}"] = (nmetrics[key] ** 2).mean().sqrt()
        return loss, metrics

    def fit(  # noqa: C901 TODO too complex
        self,
        workdir_name: str,
        device: str = "cuda",
        seed: int = 0,
        validation_set: Optional[
            Tuple[
                Basis,
                Tensor,
                Union[SCFData, Tuple[Union[float, Tensor], Tensor]],
            ]
        ] = None,
        validation_step: int = 0,
        with_adam: bool = False,
        loss_threshold: float = 0.0,
        clip_grad_norm: Optional[float] = None,
        lr: float = 1e-2,
        minibatchsize: Optional[int] = None,
        lr_scheduler: str = "ReduceLROnPlateau",
        lr_scheduler_kwargs: Dict[str, Any] = SCHEDULER_KWARGS,
        store_individual_loss: bool = False,
        **validation_kwargs: Any,
    ) -> None:
        """Execute training process of the model."""
        workdir = Path(workdir_name)
        self.minibatchsize = minibatchsize
        self.store_individual_loss = store_individual_loss
        assert self.minibatchsize is None or self.supports_minibatch
        if self.minibatchsize is not None:
            assert self.supports_minibatch
            assert type(self.data) == GradTTData or type(self.data) == GradTTDataBasis
            dataloader = DataLoader(
                self.data, batchsize=self.minibatchsize, shuffle=True
            )
        if seed is not None:
            log.info(f"Setting random seed: {seed}")
            torch.manual_seed(seed)
        writer = SummaryWriter(log_dir=workdir)
        log.info(f"Moving to device ({device})...")
        chkpt = CheckpointStore()
        validation_chkpt = CheckpointStore()
        if validation_set is not None:
            assert validation_step
            v_basis, v_occ, v_data, v_samples = self.prepare_validation_data(
                validation_set[0], validation_set[1], validation_set[2]
            )
            self.v_basis = v_basis
            self.register_buffer("v_occ", v_occ)
            self.v_data = v_data
        self.to(device)
        log.info("Initialized training")
        step = 0
        last_log = 0.0
        metrics = None
        bestv_loss = torch.inf
        with tqdm(total=self.steps, disable=None) as pbar:

            def closure() -> float:
                nonlocal step, last_log, metrics, bestv_loss
                opt.zero_grad()
                if not self.minibatchsize:
                    metrics = self.training_step(self.data)
                else:
                    _metrics = [self.training_step(data, opt) for data in dataloader]
                    metrics = {
                        k: torch.stack([m[k] for m in _metrics]).mean()
                        for k in _metrics[0]
                    }
                assert not any(v.grad_fn for v in metrics.values())
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.functional.parameters(), clip_grad_norm
                    )
                pbar.update()
                pbar.set_postfix(loss=f"{metrics['loss'].item():.2e}")
                now = time.time()
                if now - last_log > 60:
                    log.info(
                        f"Progress: {step}/{self.steps}"
                        f', loss = {metrics["loss"]:.2e}'
                    )
                    last_log = now
                    chkpt.replace(
                        self.functional.state_dict(),
                        workdir / f"chkpt-{step}.pt",
                    )
                if validation_step != 0 and step % validation_step == 0:
                    assert type(self.v_occ) is Tensor
                    self.eval()
                    v_metrics: Metrics
                    v_loss, v_metrics = self.validation_eval(
                        self.v_basis, self.v_occ, self.v_data, **validation_kwargs
                    )
                    for key in v_metrics.keys():
                        metrics[key] = v_metrics[key]
                    if v_loss < bestv_loss:
                        validation_chkpt.replace(
                            self.functional.state_dict(),
                            workdir / "best_validation.pt",
                        )
                        bestv_loss = v_loss.item()
                    self.train()
                with torch.no_grad():
                    self.after_step(step, metrics, writer)
                step += 1
                return metrics["loss"].item()

            _adam_steps = 0
            if with_adam:
                opt: torch.optim.Optimizer = torch.optim.AdamW(
                    self.functional.parameters(), lr=lr
                )
                scheduler_factory = getattr(torch.optim.lr_scheduler, lr_scheduler)
                scheduler = scheduler_factory(opt, **lr_scheduler_kwargs)
                for _adam_steps in range(self.steps):
                    if isinstance(self.basis, GaussianBasis):
                        self.basis.mask = (
                            torch.rand_like(self.basis.grid_weights) <= 0.0025
                        )
                    loss = closure()
                    lr = opt.state_dict()["param_groups"][0]["lr"]
                    writer.add_scalar("learning_rate", lr, step)
                    if not self.minibatchsize:
                        opt.step()
                    if lr_scheduler == "ReduceLROnPlateau":
                        scheduler.step(loss)
                    else:
                        scheduler.step()
                    if loss < loss_threshold:
                        break
            if self.minibatchsize is None:
                opt = torch.optim.LBFGS(
                    self.functional.parameters(),
                    line_search_fn="strong_wolfe",
                    max_eval=self.steps - _adam_steps,
                    max_iter=self.steps - _adam_steps,
                    tolerance_change=0e0,
                )
                opt.step(closure)
        torch.save(metrics, workdir / "metrics.pt")
        chkpt.replace(self.functional.state_dict(), workdir / "model.pt")
        grad_norm = self.grad_norm_fn()
        if step < self.steps and grad_norm > 1e-3:
            raise GradientError(
                "Model stopped improving but big gradient norm found:"
                + f"{grad_norm}. Probably wrong gradient was evaluated."
            )

    def after_step(
        self, step: int, metrics: Dict[str, Tensor], writer: SummaryWriter
    ) -> None:
        """After step tasks."""
        for k, v in metrics.items():
            writer.add_scalar(k, v, step)
        grad_norm = self.grad_norm_fn()
        writer.add_scalar("grad/norm", grad_norm, step)

    def grad_norm_fn(self) -> Tensor:
        return torch.cat(
            [
                p.grad.flatten()
                for p in self.functional.parameters()
                if p.grad is not None
            ]
        ).norm()


class SCFTrainingTask(TrainingTask):
    """Represents a training task involving an SCF calculation."""

    make_solver: type[SCFSolver]
    data: SCFData

    def __init__(
        self,
        functional: Functional,
        basis: Basis,
        occ: Tensor,
        data: Union[SCFData, Tuple[Union[float, Tensor], Tensor]],
        make_solver: type[SCFSolver],
        steps: int = 200,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        basis, occ, data, self.train_samples = self.prepare_validation_data(
            basis, occ, data
        )
        self.make_solver = make_solver
        self.functional = functional
        self.basis = basis
        self.register_buffer("occ", occ)
        self.data = data
        self.steps = steps
        self.kwargs = kwargs
        assert self.make_solver is not None
        self.store_individual_loss = False

    def metrics_fn(
        self,
        data: Union[SCFData, GradTTData, GradTTDataBasis],
    ) -> Metrics:
        """Evaluate model provided a basis and orbital occupation numbers.

        For a method that computes a fixed point, the gradient of the loss with
        respect to the net parameters is independent of the initial guess. To avoid
        numerical issues when backpropagating throught the SCF procedure we first
        compute the fixed point and restart the computation from a point close to
        the actual solution. This produces much more stable gradient computation.

        Ref. Deep Equilibrium Models, arXiv:1909.01377
        """
        assert type(data) == SCFData
        solver = self.make_solver(self.basis, self.occ, self.functional)
        try:
            sol_guess = solver.solve(
                create_graph=self.training,
                P_guess=data.P,
                **self.kwargs,
            )
            sol = solver.solve(
                create_graph=self.training,
                P_guess=sol_guess.P.detach()
                + (
                    torch.rand(sol_guess.P.shape[-1], device=sol_guess.P.device) * 1e-7
                ).diag(),
                **self.kwargs,
            )
            metrics = {"SCF/iter": sol_guess.niter}
        except SCFNotConvergedError as e:
            sol = e.sol
            metrics = {"SCF/iter": sol.niter}
        metrics["SCF/converged"] = sol.converged.sum()
        density_pred = self.basis.density(sol.P)
        density_true = self.basis.density(data.P)
        N = self.occ.sum(dim=-1)
        energy_loss_sq = (((sol.E - data.energy) ** 2 / N)[sol.converged]).mean()
        density_loss_sq = (
            ((self.basis.density_mse(density_pred - density_true)) / N)[sol.converged]
        ).mean()
        loss_sq = energy_loss_sq + density_loss_sq
        if self.training:
            loss_sq.backward()
        metrics["loss"] = loss_sq.detach().sqrt()
        metrics["loss/energy"] = energy_loss_sq.detach().sqrt()
        metrics["loss/regularization"] = density_loss_sq.detach().sqrt()
        return metrics


class GradientTrainingTask(TrainingTask):
    """Represents a training task with gradient regularization."""

    basis: Basis
    occ: Tensor
    data: GradTTData
    train_samples: int
    functional: Functional
    steps: int
    supports_minibatch = True

    def __init__(
        self,
        functional: Functional,
        basis: Basis,
        occ: Tensor,
        data: SCFData,
        make_solver: Optional[type[SCFSolver]] = None,
        steps: int = 200,
        l: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.functional = functional
        self.basis, occ, self.data, self.train_sample = self.prepare_data(
            basis, occ, data
        )
        self.register_buffer("occ", occ)
        self.steps = steps
        # In this training task it is not compulsory to have a solver,
        # but it is required for validation
        self.make_solver = make_solver
        self.l = l
        self.kwargs = kwargs
        self.minibatchsize = None
        self.store_individual_loss = False

    def prepare_data(
        self,
        basis: Basis,
        occ: Tensor,
        data: SCFData,
    ) -> Tuple[Basis, Tensor, GradTTData, int]:
        assert data.C is not None
        train_samples = data.energy.shape[0]
        occ_mask = torch.where(occ > 0, occ.new_ones(1), occ.new_zeros(1))
        psi = basis.get_psi(data.C)
        n = basis.density(data.P)
        # data.P.shape = [batch, (s), basis, basis]
        # P.shape = [batch, (s), (l), basis, basis]
        P = (data.C.transpose(-2, -1) * occ[..., None, :]) @ data.C
        S, T, Vext = basis.get_core_integrals()
        VH, Vfunc, Efunc = basis.get_int_integrals(data.P, self.functional)
        vext = Vext.diagonal(dim1=-1, dim2=-2)
        vH = VH.diagonal(dim1=-1, dim2=-2)
        if len(data.P.shape) == 4:  # Spin polarized case.
            P = P.sum(1)
            vext = vext[:, None, ...]
            vH = vH[:, None, ...]
        laplacian = basis.get_func_laplacian(data.C)
        energybase = ((T + Vext + 5e-1 * VH) * P).flatten(1).sum(-1) + basis.E_nuc
        TVextVH = (-0.5 * laplacian).squeeze(-1) + (vext + vH)[..., None, :] * psi
        N = occ.reshape(occ.shape[0], -1).sum(-1)
        return (
            basis,
            occ,
            GradTTData(
                N,
                occ_mask,
                psi,
                basis.grid,
                basis.grid_weights,
                basis.dv,
                n,
                TVextVH,
                energybase,
                data.energy,
            ),
            train_samples,
        )

    def metrics_fn(self, data: Union[SCFData, GradTTData, GradTTDataBasis]) -> Metrics:
        assert type(data) == GradTTData
        n = data.n.detach().requires_grad_()
        self.density = Density(n, data.grid, data.grid_weights)
        eps_func = self.functional(self.density)
        if self.functional.per_electron:
            eps_func = eps_func * self.density.density
        E_func = (eps_func * data.grid_weights * data.dv).sum(-1)
        (vfunc,) = torch.autograd.grad(
            eps_func.sum(), self.density.value, create_graph=self.training
        )

        if len(data.TVextVH.shape) == len(vfunc.shape) + 2:
            vfunc = vfunc[..., None, None, :]  # Add l and orbital dimensions
        elif len(data.TVextVH.shape) == len(vfunc.shape) + 1:
            vfunc = vfunc[..., None, :]  # Add orbital dimension
        H1 = data.TVextVH + vfunc * data.psi
        mu = (data.psi * data.grid_weights * H1).sum(-1)
        H2 = mu[..., None] * data.psi
        grad2 = data.occ_mask[..., None] * (H1 - H2) ** 2
        # grad2.shape = [batch, (spin), (l), orbital, grid]
        # data.grid_weights.shape = [(grid_batch), (grid)]
        grad2 = ((grad2.movedim(0, -2) * data.grid_weights).movedim(-2, 0)).reshape(
            grad2.shape[0], -1
        )

        energy_loss_sq = (data.Eref - (data.energybase + E_func)) ** 2 / data.N
        regularization_sq = grad2.sum(-1)

        metrics = {}
        if (
            data.Eref.shape[0] > 1
            and self.minibatchsize is None
            and self.store_individual_loss
        ):
            for i, (e, r) in enumerate(zip(energy_loss_sq, regularization_sq)):
                metrics[f"individual_loss/energy{i}"] = e.detach().sqrt()
                metrics[f"individual_loss/regularization{i}"] = r.detach().sqrt()

        energy_loss_sq = energy_loss_sq.mean()
        regularization_sq = regularization_sq.mean()
        loss_sq = energy_loss_sq + self.l * regularization_sq
        loss_sq.backward()

        metrics["loss"] = loss_sq.detach().sqrt()
        metrics["loss/energy"] = energy_loss_sq.detach().sqrt()
        metrics["loss/regularization"] = regularization_sq.detach().sqrt()
        return metrics


class GradientTrainingTaskBasis(TrainingTask):
    """Represents a training task with gradient regularization."""

    basis: Basis
    occ: Tensor
    data: GradTTDataBasis
    train_samples: int
    functional: Functional
    steps: int
    supports_minibatch = True

    def __init__(
        self,
        functional: Functional,
        basis: Basis,
        occ: Tensor,
        data: SCFData,
        make_solver: Optional[type[SCFSolver]] = None,
        steps: int = 200,
        l: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.functional = functional
        self.basis, occ, self.data, self.train_sample = self.prepare_data(
            basis, occ, data
        )
        self.register_buffer("occ", occ)
        self.steps = steps
        # In this training task it is not compulsory to have a solver,
        # but it is required for validation
        self.make_solver = make_solver
        self.l = l
        self.kwargs = kwargs
        self.minibatchsize = None
        self.store_individual_loss = False

    def prepare_data(
        self,
        basis: Basis,
        occ: Tensor,
        data: SCFData,
    ) -> Tuple[Basis, Tensor, GradTTDataBasis, int]:
        assert data.C is not None
        train_samples = data.energy.shape[0]
        occ_mask = torch.where(occ > 0, occ.new_ones(1), occ.new_zeros(1))
        P = (data.C.transpose(-2, -1) * occ[..., None, :]) @ data.C
        S, T, Vext = basis.get_core_integrals()
        VH, Vfunc, Efunc = basis.get_int_integrals(data.P, self.functional)
        if len(data.P.shape) == 4:
            T = T[:, None, ...]
            Vext = Vext[:, None, ...]
            VH = VH[:, None, ...]
        TVext = T + Vext
        energybase = ((T + Vext + 5e-1 * VH) * P).flatten(1).sum(-1) + basis.E_nuc
        N = occ.reshape(occ.shape[0], -1).sum(-1)
        return (
            basis,
            occ,
            GradTTDataBasis(
                N,
                occ_mask,
                data.C,
                S,
                TVext,
                energybase,
                data.energy,
            ),
            train_samples,
        )

    def metrics_fn(self, data: Union[SCFData, GradTTData, GradTTDataBasis]) -> Metrics:
        assert type(data) == GradTTDataBasis
        P = (data.C.transpose(-2, -1) * self.occ[..., None, :]) @ data.C
        if len(data.TVext.shape) == 5:
            P = P.sum(-3)  # Sum extra_fock_matrix dimension
        VH, Vfunc, Efunc = self.basis.get_int_integrals(
            P, self.functional, create_graph=self.training
        )
        if len(P.shape) == 4:
            VH = VH[:, None, ...]
        F = (data.TVext + VH + Vfunc).unsqueeze(-3)  # Add orbital dimension
        epsilon = (F * (data.C[..., None] * data.C[..., None, :])).sum((-1, -2))
        gi = torch.einsum(
            "...i, ...ji -> ...j", data.C, F - epsilon[..., None, None] * data.S
        )
        G = gi[..., None] * gi[..., None, :]
        gnorm = (G * data.S).flatten(1).sum(-1)

        energy_loss_sq = (data.Eref - (data.energybase + Efunc)) ** 2 / data.N
        regularization_sq = gnorm

        metrics = {}
        if (
            data.Eref.shape[0] > 1
            and self.minibatchsize is None
            and self.store_individual_loss
        ):
            for i, (e, r) in enumerate(zip(energy_loss_sq, regularization_sq)):
                metrics[f"individual_loss/energy{i}"] = e.detach().sqrt()
                metrics[f"individual_loss/regularization{i}"] = r.detach().sqrt()

        energy_loss_sq = energy_loss_sq.mean()
        regularization_sq = regularization_sq.mean()
        loss_sq = energy_loss_sq + self.l * regularization_sq
        loss_sq.backward()

        metrics["loss"] = loss_sq.detach().sqrt()
        metrics["loss/energy"] = energy_loss_sq.detach().sqrt()
        metrics["loss/regularization"] = regularization_sq.detach().sqrt()
        return metrics
