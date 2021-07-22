# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Any, Callable, Dict, List, Tuple, TypeVar, Union

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .basis import Basis
from .errors import SCFNotConvergedError
from .functional import Functional
from .scf import solve_scf
from .utils import SystemBatch

__all__ = ["train_functional"]

T_co = TypeVar("T_co", covariant=True)
LossFn = Callable[
    [Basis, Tensor, Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Dict[str, Tensor]]
]


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
) -> Tuple[Tensor, Dict[str, Tensor]]:
    E_loss = (E_pred[-1] - E_truth) ** 2 / N
    n_loss = basis.density_mse(n_pred[-1] - n_truth) / N
    loss = E_loss + n_loss
    return loss, {"n_loss": n_loss.detach(), "E_loss": E_loss.detach()}


def training_step(
    basis: Basis,
    occ: Tensor,
    functional: Functional,
    loss_fn: LossFn,
    E_truth: Tensor,
    n_truth: Tensor,
    **kwargs: Any,
) -> Tuple[Tensor, Dict[str, Tensor]]:
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
    metrics["scf_it"] = torch.tensor(len(tape))
    return loss.detach(), metrics
