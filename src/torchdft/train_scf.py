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
    List,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .basis import Basis
from .errors import SCFNotConvergedError
from .functional import Functional
from .scf import solve_scf
from .utils import SystemBatch

__all__ = ["train_functional"]

T_co = TypeVar("T_co", covariant=True)
Metrics = Dict[str, Tensor]
LossFn = Callable[
    [Basis, Tensor, Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Metrics]
]

log = logging.getLogger(__name__)


def train_functional(
    basis_class: Callable[[SystemBatch], Basis],
    functional: Functional,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader[T_co],
    checkpoint_freq: Union[bool, int] = False,
    max_epochs: int = 10,
    max_iterations: int = 15,
    mode: str = "KS",
    requires_closure: bool = False,
    writer: SummaryWriter = None,
    max_grad_norm: float = None,
    **kwargs: Any,
) -> float:
    """Train a functional."""
    assert len(dataloader) == 1
    systemlist, E_truth, n_truth = next(iter(dataloader))
    basis_list, occ_list = [], []
    for system in systemlist:
        basis_list.append(basis_class(system))
        occ_list.append(system.occ(mode=mode))
    step = 0
    for _epoch in range(max_epochs):

        def closure() -> float:
            nonlocal step
            optimizer.zero_grad()
            losses = [
                training_step(
                    basis,
                    occ,
                    functional,
                    energy_density_loss,
                    *data,
                    max_iterations=max_iterations,
                    **kwargs,
                )
                for basis, occ, *data in zip(basis_list, occ_list, E_truth, n_truth)
            ]
            loss, metrics = list(zip(*losses))
            loss = torch.stack(loss).mean()
            metrics = {k: torch.stack([m[k] for m in metrics]) for k in metrics[0]}
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm(functional.parameters(), max_grad_norm)
            if writer:
                E_loss = metrics["E_loss"].mean()
                n_loss = metrics["n_loss"].mean()
                writer.add_scalars(
                    "Losses",
                    {"E_loss": E_loss, "n_loss": n_loss, "Loss": loss},
                    step,
                )
                writer.add_scalars(
                    "SCF_iterations",
                    {
                        "max_it": metrics["scf_it"].max(),
                        "min_it": metrics["scf_it"].min(),
                        "mean_it": metrics["scf_it"].mean(),
                    },
                    step,
                )
                parameters = [p for p in functional.parameters() if p.grad is not None]
                total_norm = torch.norm(
                    torch.stack([torch.norm(p.grad.detach()) for p in parameters])
                )
                writer.add_scalars(
                    "Gradients",
                    {"grad_norm": total_norm},
                    step,
                )
            if checkpoint_freq and step % checkpoint_freq == 0:
                torch.save(functional.state_dict(), f"checkpoint_{step}.pth")
            step += 1
            return loss.item()

        if requires_closure:
            optimizer.step(closure)
        else:
            closure()
            optimizer.step()

    loss = closure()
    return loss


def energy_density_loss(
    basis: Basis,
    N: Tensor,
    E_pred: Tensor,
    n_pred: Tensor,
    E_truth: Tensor,
    n_truth: Tensor,
) -> Tuple[Tensor, Metrics]:
    E_loss_sq = ((E_pred[-1] - E_truth) ** 2 / N).mean()
    n_loss_sq = (basis.density_mse(n_pred[-1] - n_truth) / N).mean()
    loss_sq = E_loss_sq + n_loss_sq
    return loss_sq.sqrt(), {
        "n_loss": n_loss_sq.detach().sqrt(),
        "E_loss": E_loss_sq.detach().sqrt(),
    }


def training_step(
    basis: Basis,
    occ: Tensor,
    functional: Functional,
    loss_fn: LossFn,
    E_truth: Tensor,
    n_truth: Tensor,
    **kwargs: Any,
) -> Tuple[Tensor, Metrics]:
    tape: List[Tuple[Tensor, Tensor]] = []
    try:
        solve_scf(
            basis,
            occ,
            functional,
            tape=tape,
            create_graph=True,
            **kwargs,
        )
    except SCFNotConvergedError:
        pass
    n_pred, E_pred = list(zip(*tape))
    E_pred = torch.stack(E_pred)
    n_pred = basis.density(torch.stack(n_pred))
    loss, metrics = loss_fn(basis, occ.sum(dim=-1), E_pred, n_pred, E_truth, n_truth)
    assert not any(v.grad_fn for v in metrics.values())
    loss.backward()
    metrics["SCF/iter"] = torch.tensor(len(tape))
    return loss.detach(), metrics


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


class TrainingTask:
    def __init__(
        self,
        functional: Functional,
        basis: Basis,
        occ: Tensor,
        data: Union[SCFData, Tuple[Union[float, Tensor], Tensor]],
        steps: int = 200,
        **kwargs: Any,
    ) -> None:
        energy, density = data
        energy = torch.as_tensor(energy)
        self.functional = functional
        self.basis = basis
        self.occ = occ
        self.data = SCFData(energy, density)
        self.steps = steps
        self.kwargs = kwargs

    def eval_model(
        self, basis: Basis, occ: Tensor, create_graph: bool = False
    ) -> Tuple[SCFData, Metrics]:
        tape: List[Tuple[Tensor, Tensor]] = []
        try:
            solve_scf(
                basis,
                occ,
                self.functional,
                tape=tape,
                create_graph=create_graph,
                **self.kwargs,
            )
        except SCFNotConvergedError:
            pass
        metrics = {"SCF/iter": torch.tensor(len(tape))}
        n_pred, E_pred = list(zip(*tape))
        E_pred = torch.stack(E_pred)
        n_pred = basis.density(torch.stack(n_pred))
        return SCFData(E_pred, n_pred), metrics

    def metrics_fn(
        self, basis: Basis, occ: Tensor, data: SCFData, create_graph: bool = False
    ) -> Metrics:
        data_pred, metrics = self.eval_model(basis, occ, create_graph=create_graph)
        N = self.occ.sum(dim=-1)
        energy_loss_sq = ((data_pred.energy[-1] - data.energy) ** 2 / N).mean()
        density_loss_sq = (
            basis.density_mse(data_pred.density[-1] - data.density) / N
        ).mean()
        loss = (energy_loss_sq + density_loss_sq).sqrt()
        metrics["loss"] = loss
        metrics["loss/energy"] = energy_loss_sq.detach().sqrt()
        metrics["loss/density"] = density_loss_sq.detach().sqrt()
        return metrics

    def training_step(self) -> Metrics:
        metrics = self.metrics_fn(self.basis, self.occ, self.data, create_graph=True)
        loss = metrics["loss"]
        loss.backward()
        loss.detach_()
        assert not any(v.grad_fn for v in metrics.values())
        return metrics

    def train(self, workdir: str, device: str = "cuda", seed: int = 0) -> None:
        workdir = Path(workdir)
        if seed is not None:
            log.info(f"Setting random seed: {seed}")
            torch.manual_seed(seed)
        writer = SummaryWriter(log_dir=workdir)
        log.info(f"Moving to device ({device})...")
        self.functional.to(device)
        self.basis.to(device)
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

    def after_step(
        self, step: int, metrics: Dict[str, Tensor], writer: SummaryWriter
    ) -> None:
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
