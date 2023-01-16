from typing import Dict, Mapping

import pyspiel

from cfr_discounted2 import DiscountedCFR2
from type_aliases import Action, Infostate, Probability, Regret, Value


class LinearCFR2(DiscountedCFR2):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            dict(
                alpha=1.0,
                beta=1.0,
                gamma=1.0,
            )
        )
        super().__init__(*args, **kwargs)
