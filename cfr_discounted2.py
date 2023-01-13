from typing import Dict, Mapping

import pyspiel

import rm
from cfr2 import CFR2
from type_aliases import Action, Infostate, Probability, Regret, Value

from cfr_discounted import DiscountedCFR


class DiscountedCFR2(CFR2):
    def __init__(
        self,
        *args,
        alpha: float = 3.0 / 2.0,
        beta: float = 0.0,
        gamma: float = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _average_policy_at(self, current_player, infostate):
        gamma_weight = (self.iteration / (self.iteration + 1)) ** self.gamma
        # this function is always called before the base class updates its average policy,
        # so it is the right time to apply the previous iterations discount on it
        player_policy = super()._average_policy_at(current_player, infostate)
        for action in player_policy.keys():
            player_policy[action] *= gamma_weight
        return player_policy
