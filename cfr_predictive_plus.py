from collections.abc import MutableMapping
from enum import Enum
from typing import List, Dict, Sequence

from cfr_discounted import DiscountedCFR2
from type_aliases import Infostate, Value, Action


class Players(Enum):
    chance = -1
    player1 = 0
    player2 = 1


class PredictivePlusCFR(DiscountedCFR2):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        kwargs["simultaneous_updates"] = False
        super().__init__(
            *args, **kwargs, alpha=float("inf"), beta=-float("inf"), gamma=2
        )
        self._action_values: Dict[Infostate, MutableMapping[Action, Value]] = dict()

    def _action_value_map(self, infostate: Infostate):
        # reset the action values found previously
        self._action_values[infostate] = {}
        return self._action_values[infostate]

