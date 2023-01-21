from collections import defaultdict, deque
from copy import deepcopy
from typing import Dict, Mapping, Optional, Type, Sequence, MutableMapping

from cfr2_base import CFRBase
from type_aliases import Action, Infostate, Probability, Value
import numpy as np

import rm
from rm import ExternalRegretMinimizer
import pyspiel

from utils import (
    counterfactual_reach_prob,
    print_final_policy_profile,
    print_kuhn_poker_policy_profile,
    to_pyspiel_tab_policy,
    normalize_policy_profile,
)


class CFR2(CFRBase):
    def iterate(
        self,
        traversing_player: Optional[int] = None,
    ):
        traversing_player = self._cycle_updating_player(traversing_player)

        if self._verbose:
            print(
                "\nIteration",
                self._alternating_update_msg() if self.alternating else self.iteration,
            )
        self._traverse(
            self.root_state.clone(),
            reach_prob_map={player: 1.0 for player in [-1] + self.players},
            traversing_player=traversing_player,
        )
        self._iteration += 1

    def _traverse(
        self,
        state: pyspiel.State,
        reach_prob_map: dict[Action, Probability],
        traversing_player: Optional[int] = None,
    ):
        if state.is_terminal():
            return state.returns()

        if state.is_chance_node():
            return self._traverse_chance_node(state, reach_prob_map, traversing_player)
        else:
            curr_player = state.current_player()
            infostate = state.information_state_string(curr_player)

            action_values = self._action_value_map(infostate)
            state_value = self._traverse_player_node(
                state, infostate, reach_prob_map, traversing_player, action_values
            )
            if self.simultaneous or traversing_player == curr_player:
                regret_minimizer = self.regret_minimizer(infostate)
                self._update_regret(
                    regret_minimizer,
                    action_values,
                    state_value,
                    reach_prob_map,
                    curr_player,
                )
                self._update_avg_policy(
                    regret_minimizer.recommend(self.iteration),
                    reach_prob_map,
                    infostate,
                    curr_player,
                )
            return state_value

    def _traverse_chance_node(self, state, reach_prob, updating_player):
        state_value = np.zeros(len(self.players))
        for outcome, outcome_prob in state.chance_outcomes():
            next_state = state.child(outcome)

            child_reach_prob = deepcopy(reach_prob)
            child_reach_prob[state.current_player()] *= outcome_prob

            action_value = self._traverse(
                next_state, child_reach_prob, updating_player
            )
            state_value += outcome_prob * np.asarray(action_value)
        return state_value

    def _traverse_player_node(
        self, state, infostate, reach_prob, updating_player, action_values
    ):
        current_player = state.current_player()
        state_value = np.zeros(len(self.players))

        self._set_action_list(infostate, state)
        regret_minimizer = self.regret_minimizer(infostate)
        current_policy = regret_minimizer.recommend(self.iteration)

        for action, action_prob in current_policy.items():
            child_reach_prob = deepcopy(reach_prob)
            child_reach_prob[current_player] *= action_prob
            next_state = state.child(action)

            action_values[action] = self._traverse(
                next_state, child_reach_prob, updating_player
            )
            state_value += action_prob * np.asarray(action_values[action])

        return state_value

    def _update_regret(
        self,
        regret_minimizer: ExternalRegretMinimizer,
        action_values: Mapping[Action, Sequence[Value]],
        state_value: Sequence[Value],
        reach_probs: Dict[int, Probability],
        curr_player: int,
    ):
        player_state_value = state_value[curr_player]
        cf_reach_p = counterfactual_reach_prob(reach_probs, curr_player)
        regret_minimizer.observe_regret(
            self.iteration,
            lambda a: cf_reach_p * (action_values[a][curr_player] - player_state_value),
        )

    def _update_avg_policy(
        self,
        curr_policy: Dict[Action, Probability],
        reach_prob: Dict[int, Probability],
        infostate: Infostate,
        curr_player: int,
    ):
        player_reach_prob = reach_prob[curr_player]
        avg_policy = self._avg_policy_at(curr_player, infostate)
        for action, curr_policy_prob in curr_policy.items():
            avg_policy[action] += player_reach_prob * curr_policy_prob
