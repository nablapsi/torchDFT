# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from abc import abstractmethod


class Basis:
    """Base class representing an abstract basis."""

    @abstractmethod
    def get_core_integrals(self):
        pass

    @abstractmethod
    def get_int_integrals(self, P, xc_functional):
        pass
