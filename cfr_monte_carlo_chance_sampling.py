from typing import Optional, Union

import numpy as np

from cfr import CFR


class ChanceSamplingCFR(CFR):
    def __init__(
        self, *args, seed: Optional[Union[int, np.random.Generator]] = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.rng: np.random.Generator = np.random.default_rng(seed)

    def _traverse_chance_node(self, state, reach_prob, updating_player):
        outcome, outcome_prob = self.rng.choice(state.chance_outcomes())
        state.apply_action(int(outcome))
        return self._traverse(state, reach_prob, updating_player)
