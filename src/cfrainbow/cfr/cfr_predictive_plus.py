from collections import defaultdict
from collections.abc import MutableMapping
from copy import deepcopy, copy
from dataclasses import astuple, dataclass
from typing import Dict, Sequence, Optional, Type, List, Set

import pyspiel

from cfrainbow.rm import ExternalRegretMinimizer
from cfrainbow.spiel_types import Infostate, Value, Action, Probability, Regret, Player
from cfrainbow.utils import counterfactual_reach_prob, infostates_gen
from .cfr_base import iterate_logging
from .cfr_discounted import DiscountedCFR


@dataclass
class ActionValues:
    __slots__ = ["mapping", "cf_reach_probability", "iteration"]
    mapping: Dict[Action, Value]
    cf_reach_probability: float
    iteration: int


class GetitemZero:
    def __getitem__(self, item):
        return 0.0


class PredictivePlusCFR(DiscountedCFR):
    def __init__(
        self,
        *args,
        alpha=float("inf"),
        beta=-float("inf"),
        gamma=1,
        **kwargs,
    ):
        kwargs.update(
            dict(
                # only allow alternating update cycles
                alternating=True,
            )
        )
        super().__init__(
            *args,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            **kwargs,
        )
        self._current_action_values: Dict[Infostate, ActionValues] = defaultdict(
            lambda: ActionValues(defaultdict(float), 0.0, -1)
        )

        self._prev_action_values: Dict[Infostate, ActionValues] = copy(
            self._current_action_values
        )

        self.infostates = {player: set() for player in self.players}
        for infostate, player, state, _ in infostates_gen(root=self.root_state.clone()):
            self.infostates[player].add(infostate)

    @iterate_logging
    def iterate(
        self,
        updating_player: Optional[int] = None,
    ):
        updating_player = self._cycle_updating_player(updating_player)
        self._traverse(
            self.root_state.clone(),
            reach_prob_map={player: 1.0 for player in [-1] + self.players},
            updating_player=updating_player,
        )
        self._move_curr_to_prev(updating_player)
        self._iteration += 1

    def utility_prediction(self, infostate: Infostate):
        """Returns the current prediction of next iteration's payoff

        Parameters
        ----------
        infostate: Optional[Infostate]
            the infostate at which to predict future payoffs

        Returns
        -------
        Dict[Action, float]
            the action value predictions for the next turns.
        """
        if (self.iteration - self.nr_players) < 0:
            return GetitemZero()
        return self._prev_action_values[infostate].mapping
        # av = self._prev_action_values[infostate]
        # print(
        #     "iteration",
        #     self.iteration,
        #     "fetches prediction from iteration",
        #     av.iteration,
        # )
        # return {
        #     # action: value / (av.cf_reach_probability + 1e-32)
        #     action: value
        #     for action, value in av.mapping.items()
        # }

    def _traverse_player_node(
        self, state, infostate, reach_prob, updating_player, action_values
    ):
        current_player = state.current_player()
        state_value = [0.0] * self.nr_players
        av = self._current_action_values[infostate]
        cf_rp = counterfactual_reach_prob(reach_prob, current_player)
        av.cf_reach_probability += cf_rp
        # for action, action_prob in (
        #     self.regret_minimizer(infostate)
        #     .recommend(self.iteration, prediction=self.utility_prediction(infostate))
        #     .items()
        # ):
        for action, action_prob in (
            self.regret_minimizer(infostate).recommend(self.iteration).items()
        ):
            child_value = self._traverse(
                state.child(action),
                self.child_reach_prob_map(reach_prob, current_player, action_prob),
                updating_player,
            )
            action_values[action] = child_value

            for p in self.players:
                state_value[p] += action_prob * child_value[p]
            av.mapping[action] += cf_rp * child_value[current_player]

        return state_value

    def _move_curr_to_prev(self, updating_player):
        for infostate in self.infostates[updating_player]:
            # if self.verbose:
            #     print("Forcing the update on infostate", infostate)
            # self.regret_minimizer(infostate).recommend(
            #     self.iteration + 1, prediction=self.utility_prediction(infostate), force=True
            # )
            curr_av = self._current_action_values[infostate]
            curr_av.iteration = self.iteration
            self._prev_action_values[infostate] = curr_av
        self._current_action_values.clear()
