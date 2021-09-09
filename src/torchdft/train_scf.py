# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .basis import Basis
from .errors import SCFNotConvergedError
from .functional import Functional
from .scf import solve_scf

__all__ = ["TrainingTask"]

Metrics = Dict[str, Tensor]
LossFn = Callable[
    [Basis, Tensor, Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Metrics]
]

log = logging.getLogger(__name__)


class SCFData(NamedTuple):
    energy: Tensor
    density: Tensor

    def to(self, device: Optional[str]) -> SCFData:
        return SCFData(self.energy.to(device), self.density.to(device))


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
        energy, density = data
        energy = torch.as_tensor(energy)
        self.functional = functional
        self.occ = occ
        self.steps = steps
        self.kwargs = kwargs
        if isinstance(basis, Basis):
            self.basislist = [basis]
            self.occ = self.occ.reshape([1, -1])
            self.data = SCFData(energy.reshape([1, 1]), density.reshape([1, -1]))
        else:
            self.basislist = list(basis)
            self.data = SCFData(energy, density)
        assert (
            len(self.basislist)
            == self.occ.shape[0]
            == self.data.energy.shape[0]
            == self.data.density.shape[0]
        )

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

    def metrics_fn(self, basis: Basis, occ: Tensor, data: SCFData) -> Metrics:
        """Evaluate the losses on current model."""
        data_pred, metrics = self.eval_model(basis, occ)
        N = occ.sum(dim=-1)
        energy_loss_sq = ((data_pred.energy[-1] - data.energy) ** 2 / N).mean()
        density_loss_sq = (
            basis.density_mse(data_pred.density[-1] - data.density) / N
        ).mean()
        loss = energy_loss_sq + density_loss_sq
        if self.training:
            loss.backward()
        metrics["loss"] = loss.detach().sqrt()
        metrics["loss/energy"] = energy_loss_sq.detach().sqrt()
        metrics["loss/density"] = density_loss_sq.detach().sqrt()
        return metrics

    def training_step(self) -> Metrics:
        """Execute a training step."""
        assert self.training
        metrics = [
            self.metrics_fn(basis, occ, SCFData(*data))
            for basis, occ, *data in zip(
                self.basislist, self.occ, self.data.energy, self.data.density
            )
        ]
        metrics = {k: torch.stack([m[k] for m in metrics]) for k in metrics[0]}
        metrics["loss/energy"] = (metrics["loss/energy"] ** 2).mean()
        metrics["loss/density"] = (metrics["loss/density"] ** 2).mean()
        metrics["loss"] = (metrics["loss/energy"] + metrics["loss/density"]).sqrt()
        metrics["loss/energy"] = metrics["loss/energy"].sqrt()
        metrics["loss/density"] = metrics["loss/density"].sqrt()
        metrics["SCF/iter"] = max(metrics["SCF/iter"])
        # Evaluate (d RMSE / d theta) from (d MSE / d theta)
        for p in self.functional.parameters():
            p.grad = p.grad / (2.0 * metrics["loss"])
        assert not any(v.grad_fn for v in metrics.values())
        return metrics

    def fit(self, workdir: str, device: str = "cuda", seed: int = 0) -> None:
        """Execute training process of the model."""
        workdir = Path(workdir)
        if seed is not None:
            log.info(f"Setting random seed: {seed}")
            torch.manual_seed(seed)
        writer = SummaryWriter(log_dir=workdir)
        log.info(f"Moving to device ({device})...")
        self.functional.to(device)
        self.basislist = [basis.to(device) for basis in self.basislist]
        self.occ = self.occ.to(device)
        self.data = self.data.to(device)
        chkpt = CheckpointStore()
        opt = torch.optim.LBFGS(
            self.functional.parameters(),
            line_search_fn="strong_wolfe",
            max_eval=self.steps,
            max_iter=self.steps,
            tolerance_change=np.nan,
        )
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
                with torch.no_grad():
                    self.after_step(step, metrics, writer)
                step += 1
                return metrics["loss"].item()

            opt.step(closure)

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
