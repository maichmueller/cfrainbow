from copy import deepcopy
from typing import Dict, Mapping, Optional, Type

import pyspiel

from cfrainbow.rm import ExternalRegretMinimizer, InternalRegretMinimizer
from cfrainbow.spiel_types import Action, Infostate, Probability
from cfrainbow.utils import counterfactual_reach_prob, sample_on_policy

from . import PureCFR
from .cfr_base import iterate_logging
import logging


class InternalCFR(PureCFR):
    def __init__(
        self,
        root_state: pyspiel.State,
        external_regret_minimizer_type: Type[ExternalRegretMinimizer],
        internal_regret_minimizer_type: Type[InternalRegretMinimizer],
        external_rm_kwargs: Mapping,
        internal_rm_kwargs: Mapping,
        **kwargs,
    ):
        super().__init__(
            root_state, external_regret_minimizer_type, **external_rm_kwargs, **kwargs
        )
        logging.warning("Internal CFR is not yet implemented.")
        self._internal_regret_minimizer_type: Type[
            InternalRegretMinimizer
        ] = internal_regret_minimizer_type
        self._internal_regret_minimizer_kwargs: Mapping = internal_rm_kwargs
        self.plan: Dict[Infostate, Action] = {}

    @iterate_logging
    def iterate(
        self,
        updating_player: Optional[int] = None,
    ):
        # empty the previously sampled strategy
        self.plan.clear()

        self._traverse(
            self.root_state.clone(),
            reach_prob_map=(
                {player: 1.0 for player in [-1] + self.players}
                if self.simultaneous
                else None
            ),
            updating_player=self._cycle_updating_player(updating_player),
        )
        self._iteration += 1

    def _traverse(
        self,
        state: pyspiel.State,
        reach_prob_map: Optional[Dict[int, Probability]] = None,
        updating_player: Optional[int] = None,
    ):
        if state.is_terminal():
            return state.returns()

        if state.is_chance_node():
            return self._traverse_chance_node(state, reach_prob_map, updating_player)

        curr_player = state.current_player()
        infostate = state.information_state_string(curr_player)
        self._set_action_list(infostate, state)
        actions = self.action_list(infostate)
        regret_minimizer = self.regret_minimizer(infostate)
        player_policy = regret_minimizer.recommend(self.iteration)

        sampled_action = self._sample_action(infostate, player_policy)
