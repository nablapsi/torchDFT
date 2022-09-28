# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import typing

if typing.TYPE_CHECKING:
    from .scf import SCFSolution

__all__ = ()


class TorchDFTError(Exception):
    pass


class SCFNotConvergedError(TorchDFTError):
    def __init__(self, sol: "SCFSolution") -> None:
        self.sol = sol


class NanError(TorchDFTError):
    pass


class GradientError(TorchDFTError):
    pass
