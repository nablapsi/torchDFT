# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Any, Callable, Dict, List, Tuple, TypeVar, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .basis import Basis
from .errors import SCFNotConverged
from .functional import Functional
from .scf import solve_scf
from .utils import SystemBatch

__all__ = ["train_functional"]

T_co = TypeVar("T_co", covariant=True)


def train_functional(
    basis_class: Callable[[SystemBatch], Basis],
    functional: Functional,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader[T_co],
    alpha_decay: float = 0.9,
    checkpoint_freq: Union[bool, int] = False,
    max_epochs: int = 10,
    max_iterations: int = 15,
    mode: str = "KS",
    requires_closure: bool = False,
    writer: SummaryWriter = None,
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
    for epoch in range(max_epochs):

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
            loss, E_loss, n_loss = (torch.stack(x).mean() for x in zip(*losses))
            if writer:
                writer.add_scalars(
                    "Losses",
                    {"E_loss": E_loss, "n_loss": n_loss, "Loss": loss},
                    step,
                )
            if checkpoint_freq and step % checkpoint_freq == 0:
                torch.save(functional.state_dict(), f"checkpoint_{epoch}.pth")
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
    trajectory_discount: float = 0.8,
    **kwargs: Any,
) -> Tuple[Tensor, Tensor, Tensor]:
    log_dict: Dict[str, List[Tensor]] = {}
    try:
        solve_scf(
            basis,
            occ,
            functional,
            log_dict=log_dict,
            create_graph=True,
            **kwargs,
        )
    except SCFNotConverged:
        pass
    E_pred = log_dict["energy"][-1]
    n_pred = basis.density(log_dict["denmat"][-1])
    N = occ.sum()
    E_loss = ((E_pred - E_truth) ** 2).sum(-1) / N
    n_loss = basis.density_mse(n_pred - n_truth) / N
    loss = E_loss + n_loss
    loss.backward()
    return loss, E_loss, n_loss
