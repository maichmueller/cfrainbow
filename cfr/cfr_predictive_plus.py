from collections import defaultdict
from collections.abc import MutableMapping
from enum import Enum
from typing import List, Dict, Sequence, Optional, Mapping

import numpy as np
import pyspiel

from .cfr_discounted import DiscountedCFR
from spiel_types import Infostate, Value, Action

from utils import counterfactual_reach_prob


class PredictivePlusCFR(DiscountedCFR):
    def __init__(
        self,
        *args,
        alpha=float("inf"),
        beta=-float("inf"),
        gamma=2,
        **kwargs,
    ):
        kwargs.update(
            dict(
                # only allow alternating update cycles
                alternating=True,
            )
        )
        super().__init__(*args, alpha=alpha, beta=beta, gamma=gamma, **kwargs)
        self._action_values: Dict[
            Infostate, MutableMapping[Action, Sequence[Value]]
        ] = defaultdict(dict)

    def _action_value_map(self, infostate: Optional[Infostate] = None):
        if infostate is None:
            raise ValueError(
                "Predictive CFR needs the information state to return the correct action value map."
            )
        av = self._action_values[infostate]
        # reset the action values found previously
        new_entries = {a: np.zeros(self.nr_players) for a in av}
        av.clear()
        av.update(new_entries)
        return av

    def _set_action_list(self, infostate: Infostate, state: pyspiel.State):
        if infostate not in self._action_set:
            actions = state.legal_actions()
            self._action_set[infostate] = actions
            av = self._action_values[infostate]
            for action in actions:
                av[action] = np.zeros(self.nr_players)

    def value_prediction(
        self,
        infostate: Infostate,
        reach_prob: Mapping[int, float],
        traversing_player: int,
        *args,
        **kwargs,
    ):
        """Returns the current prediction of next iteration's payoff

        Parameters
        ----------
        infostate: Optional[Infostate]
            the infostate at which to predict future payoffs
        reach_prob: Optional[Mapping[int, float]]
            the reach probabilities for each player to this node
        traversing_player: int
            the player that currently traverses the tree

        Returns
        -------
        Dict[Action, float]
            the action value predictions for the next turns.
        """
        cf_rp = counterfactual_reach_prob(reach_prob, traversing_player)
        return {
            action: cf_rp * value[traversing_player]
            for action, value in self._action_values[infostate].items()
        }
