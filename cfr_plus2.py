from typing import Dict, Mapping

import pyspiel

import rm
from cfr2 import CFR2
from type_aliases import Action, Infostate, Probability, Regret, Value

from cfr_discounted2 import DiscountedCFR2


class CFRPlus2(DiscountedCFR2):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        kwargs["simultaneous_updates"] = False
        super().__init__(
            *args, **kwargs, alpha=float("inf"), beta=-float("inf"), gamma=1
        )
