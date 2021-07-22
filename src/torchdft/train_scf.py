# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Any, Callable, Dict, Tuple, TypeVar, Union

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


def train_functional(
    basis_class: Callable[[SystemBatch], Basis],
    functional: Functional,
    trainable_functional: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader[T_co],
    alpha_decay: float = 0.9,
    checkpoint_freq: Union[bool, int] = False,
    max_epochs: int = 10,
    max_iterations: int = 15,
    mode: str = "KS",
    requires_closure: bool = False,
    writer: SummaryWriter = None,
    max_grad_norm: float = 0e0,
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
                    *data,
                    max_iterations=max_iterations,
                    alpha_decay=alpha_decay,
                    **kwargs,
                )
                for basis, occ, *data in zip(basis_list, occ_list, E_truth, n_truth)
            ]
            losses = list(zip(*losses))
            loss, E_loss, n_loss, scf_it_mean = (torch.stack(x).mean() for x in losses)
            scf_it = losses[-1]
            if max_grad_norm > 0e0:
                nn.utils.clip_grad_norm(
                    trainable_functional.parameters(), max_grad_norm
                )
            if writer:
                writer.add_scalars(
                    "Losses",
                    {"E_loss": E_loss, "n_loss": n_loss, "Loss": loss},
                    step,
                )
                writer.add_scalars(
                    "SCF_iterations",
                    {
                        "max_it": max(scf_it),
                        "min_it": min(scf_it),
                        "mean_it": scf_it_mean,
                    },
                    step,
                )
            if checkpoint_freq and step % checkpoint_freq == 0:
                torch.save(trainable_functional.state_dict(), f"checkpoint_{step}.pth")
            step += 1
            return loss.item()

        if requires_closure:
            optimizer.step(closure)
        else:
            closure()
            optimizer.step()

    loss = closure()
    return loss


def training_step(
    basis: Basis,
    occ: Tensor,
    functional: Functional,
    E_truth: Tensor,
    n_truth: Tensor,
    **kwargs: Any,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    log_dict: Dict[str, Tensor] = {}
    try:
        solve_scf(
            basis,
            occ,
            functional,
            log_dict=log_dict,
            create_graph=True,
            **kwargs,
        )
    except SCFNotConvergedError:
        pass
    E_pred = log_dict["energy"]
    n_pred = basis.density(log_dict["denmat"])
    scf_it = log_dict["scf_it"]
    N = occ.sum()
    E_loss = ((E_pred - E_truth) ** 2) / N
    n_loss = basis.density_mse(n_pred - n_truth) / N
    loss = E_loss + n_loss
    loss.backward()
    return loss.detach(), E_loss.detach(), n_loss.detach(), scf_it
