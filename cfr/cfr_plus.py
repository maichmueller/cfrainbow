from typing import Type

import pyspiel

import rm
from .cfr_discounted import DiscountedCFR


class CFRPlus(DiscountedCFR):
    def __init__(
        self,
        root_state: pyspiel.State,
        regret_minimizer_type: Type[rm.ExternalRegretMinimizer] = rm.RegretMatcherPlus,
        *args,
        **kwargs,
    ):
        kwargs.update(
            dict(
                alternating=True,
                alpha=float("inf"),
                beta=-float("inf"),
                gamma=1.0,
            )
        )
        super().__init__(
            root_state,
            regret_minimizer_type,
            *args,
            **kwargs,
        )
