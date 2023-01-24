from typing import Dict, Mapping

import pyspiel

from cfr_discounted import DiscountedCFR
from type_aliases import Action, Infostate, Probability, Regret, Value


class LinearCFR(DiscountedCFR):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            dict(
                alpha=1.0,
                beta=1.0,
                gamma=1.0,
            )
        )
        super().__init__(*args, **kwargs)
