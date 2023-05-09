from typing import Optional, Union

import numpy as np

from .cfr_vanilla import VanillaCFR


class ChanceSamplingCFR(VanillaCFR):
    def _traverse_chance_node(self, state, reach_prob, updating_player):
        outcome, outcome_prob = self.rng.choice(state.chance_outcomes())
        state.apply_action(int(outcome))
        return self._traverse(state, reach_prob, updating_player)
