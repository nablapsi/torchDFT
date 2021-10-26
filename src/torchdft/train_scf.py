# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import torch
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .basis import Basis
from .errors import SCFNotConvergedError
from .functional import Functional
from .gaussbasis import GaussianBasis
from .scf import solve_scf

__all__ = ["TrainingTask"]

Metrics = Dict[str, Tensor]
LossFn = Callable[
    [Basis, Tensor, Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Metrics]
]

log = logging.getLogger(__name__)


class SCFData(nn.Module):
    energy: Tensor
    density: Tensor

    def __init__(self, energy: Tensor, density: Tensor):
        super().__init__()
        self.register_buffer("energy", energy)
        self.register_buffer("density", density)


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
        torch.save(state, self.chkpts[-1])
        while len(self.chkpts) > 1:
            self.chkpts.pop(0).unlink()


class TrainingTask(nn.Module):
    """Represents a training task."""

    basis: Union[Basis, nn.ModuleList]
    occ: Tensor
    data: SCFData

    def __init__(
        self,
        functional: Functional,
        basis: Union[Basis, Iterable[Basis]],
        occ: Tensor,
        data: Union[SCFData, Tuple[Union[float, Tensor], Tensor]],
        steps: int = 200,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        basis, occ, data, self.train_samples = self.prepare_data(basis, occ, data)
        self.functional = functional
        self.basis = basis
        self.register_buffer("occ", occ)
        self.data = data
        self.steps = steps
        self.kwargs = kwargs

    def prepare_data(
        self,
        basis: Union[Basis, Iterable[Basis]],
        occ: Tensor,
        data: Union[SCFData, Tuple[Union[float, Tensor], Tensor]],
    ) -> Tuple[Union[Basis, nn.ModuleList], Tensor, SCFData, int]:
        if not isinstance(data, SCFData):
            energy, density = data
            energy = torch.as_tensor(energy)
            data = SCFData(energy, density)
        if isinstance(basis, Basis) and not basis.E_nuc.shape:  # single basis
            samples = 1
            assert len(occ.shape) == 1
            assert len(data.energy.shape) == 0
            assert len(data.density.shape) == 1
        else:
            if isinstance(basis, Basis):  # batched basis
                assert len(basis.E_nuc.shape) == 1
                samples = basis.E_nuc.shape[0]
            else:  # iterable of bases
                basis = nn.ModuleList(list(basis))
                samples = len(basis)
            assert len(occ.shape) == 2
            if occ.shape[0] == 1:
                occ = occ.expand(samples, -1)
            assert len(data.energy.shape) == 1
            assert len(data.density.shape) == 2
            assert occ.shape[0] == samples
            assert data.energy.shape[0] == samples
            assert data.density.shape[0] == samples
        return basis, occ, data, samples

    def eval_model(self, basis: Basis, occ: Tensor) -> Tuple[SCFData, Metrics]:
        """Evaluate model provided a basis and orbital occupation numbers."""
        tape: List[Tuple[Tensor, Tensor]] = []
        try:
            solve_scf(
                basis,
                occ,
                self.functional,
                tape=tape,
                create_graph=self.training,
                **self.kwargs,
            )
        except SCFNotConvergedError:
            pass
        metrics = {"SCF/iter": torch.tensor(len(tape))}
        n_pred, E_pred = list(zip(*tape))
        E_pred = torch.stack(E_pred)
        n_pred = basis.density(torch.stack(n_pred))
        return SCFData(E_pred, n_pred), metrics

    def _metrics_fn(self, basis: Basis, occ: Tensor, data: SCFData) -> Metrics:
        data_pred, metrics = self.eval_model(basis, occ)
        N = occ.sum(dim=-1)
        energy_loss_sq = ((data_pred.energy[-1] - data.energy) ** 2 / N).mean()
        density_loss_sq = (
            basis.density_mse(data_pred.density[-1] - data.density) / N
        ).mean()
        loss_sq = energy_loss_sq + density_loss_sq
        if self.training:
            loss_sq.backward()
        metrics["loss"] = loss_sq.detach().sqrt()
        metrics["loss/energy"] = energy_loss_sq.detach().sqrt()
        metrics["loss/density"] = density_loss_sq.detach().sqrt()
        metrics.update(basis.density_metrics_fn(data_pred.density[-1], data.density))
        return metrics

    def metrics_fn(
        self,
        basis: Union[Basis, nn.ModuleList],
        occ: Tensor,
        data: SCFData,
    ) -> Metrics:
        """Evaluate the losses on current model."""
        if isinstance(basis, Basis):
            return self._metrics_fn(basis, occ, data)
        metrics = [
            self._metrics_fn(basis, occ, SCFData(*data))
            for basis, occ, *data in zip(basis, occ, data.energy, data.density)
        ]
        metrics = {k: torch.stack([m[k] for m in metrics]) for k in metrics[0]}
        metrics["loss/energy"] = (metrics["loss/energy"] ** 2).mean().sqrt()
        metrics["loss/density"] = (metrics["loss/density"] ** 2).mean().sqrt()
        metrics["loss"] = (
            metrics["loss/energy"] ** 2 + metrics["loss/density"] ** 2
        ).sqrt()
        metrics["SCF/iter"] = max(metrics["SCF/iter"])
        metrics["loss/quadrupole"] = (metrics["loss/quadrupole"] ** 2).mean().sqrt()
        return metrics

    def training_step(self) -> Metrics:
        """Execute a training step."""
        assert self.training
        metrics = self.metrics_fn(self.basis, self.occ, self.data)
        # Evaluate (d RMSE / d theta) from (d MSE / d theta)
        for p in self.functional.parameters():
            p.grad = p.grad / (2.0 * self.train_samples * metrics["loss"])
        assert not any(v.grad_fn for v in metrics.values())
        return metrics

    def fit(
        self,
        workdir: str,
        device: str = "cuda",
        seed: int = 0,
        with_lbfgs: bool = True,
        validation_set: Tuple[
            Union[Basis, Iterable[Basis]],
            Tensor,
            Union[SCFData, Tuple[Union[float, Tensor], Tensor]],
        ] = None,
        validation_step: int = 0,
    ) -> None:
        """Execute training process of the model."""
        workdir = Path(workdir)
        if seed is not None:
            log.info(f"Setting random seed: {seed}")
            torch.manual_seed(seed)
        writer = SummaryWriter(log_dir=workdir)
        log.info(f"Moving to device ({device})...")
        self.to(device)
        chkpt = CheckpointStore()
        if with_lbfgs:
            opt: torch.optim.Optimizer = torch.optim.LBFGS(
                self.functional.parameters(),
                line_search_fn="strong_wolfe",
                max_eval=self.steps,
                max_iter=self.steps,
                tolerance_change=0e0,
            )
        else:
            opt = torch.optim.AdamW(self.functional.parameters(), lr=1e-2)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=100)
        if validation_set is not None:
            assert validation_step
            v_basis, v_occ, v_data, v_samples = self.prepare_data(
                validation_set[0], validation_set[1], validation_set[2]
            )
            v_basis.to(device)
            v_occ = v_occ.to(device)
            v_data.to(device)
        log.info("Initialized training")
        step = 0
        last_log = 0.0
        metrics = None
        with tqdm(total=self.steps, disable=None) as pbar:

            def closure() -> float:
                nonlocal step, last_log, metrics
                opt.zero_grad()
                metrics = self.training_step()
                assert not any(v.grad_fn for v in metrics.values())
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
                        workdir / f"chkpt-{step}.pt",  # type: ignore
                    )
                if validation_step != 0 and step % validation_step == 0:
                    self.eval()
                    v_metrics: Metrics = self.metrics_fn(v_basis, v_occ, v_data)
                    v_metrics["loss/loss"] = v_metrics.pop("loss")
                    for key in v_metrics.keys():
                        metrics[key + "_validation"] = v_metrics[key]
                    self.train()
                with torch.no_grad():
                    self.after_step(step, metrics, writer)
                step += 1
                return metrics["loss"].item()

            if with_lbfgs:
                opt.step(closure)
            else:
                for _ in range(self.steps):
                    if isinstance(self.basis, GaussianBasis):
                        self.basis.mask = (
                            torch.rand_like(self.basis.grid_weights) <= 0.0025
                        )
                    loss = closure()
                    lr = opt.state_dict()["param_groups"][0]["lr"]
                    writer.add_scalar("learning_rate", lr, step)
                    opt.step()
                    scheduler.step(loss)

        torch.save(metrics, workdir / "metrics.pt")
        chkpt.replace(self.functional.state_dict(), workdir / "model.pt")

    def after_step(
        self, step: int, metrics: Dict[str, Tensor], writer: SummaryWriter
    ) -> None:
        """After step tasks."""
        for k, v in metrics.items():
            writer.add_scalar(k, v, step)
        grad_norm = torch.cat(
            [
                p.grad.flatten()
                for p in self.functional.parameters()
                if p.grad is not None
            ]
        ).norm()
        writer.add_scalar("grad/norm", grad_norm, step)
