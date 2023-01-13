from typing import Dict, Mapping

import pyspiel

from type_aliases import Action, Infostate, Probability, Regret, Value

from cfr_discounted import DiscountedCFR


class LinearCFR(DiscountedCFR):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            alpha=1.0,
            beta=1.0,
            gamma=1.0,
        )
