from collections import defaultdict
from copy import deepcopy
from typing import Dict, Mapping, Optional

import numpy as np

from rm import regret_matching
import pyspiel
from utils import counterfactual_reach_prob

from type_aliases import Action, Infostate, Probability, Regret, Value


class CFR:
    def __init__(
        self,
        root_state: pyspiel.State,
        curr_policy_list: list[Dict[Infostate, Mapping[Action, Probability]]],
        average_policy_list: list[Dict[Infostate, Mapping[Action, Probability]]],
        *,
        simultaneous_updates: bool = True,
        verbose: bool = False,
    ):
        self.root_state = root_state
        self.n_players = list(range(root_state.num_players()))
        self.regret_table: list[Dict[Infostate, Dict[Action, Regret]]] = [
            {} for _ in self.n_players
        ]
        self.curr_policy = curr_policy_list
        self.avg_policy = average_policy_list
        self.iteration = 0
        self._simultaneous_updates = simultaneous_updates
        self._verbose = verbose

    def average_policy(self, player: Optional[int] = None):
        if player is None:
            return self.avg_policy
        else:
            return [self.avg_policy[player]]

    def _get_current_policy(self, current_player, infostate):
        return self.curr_policy[current_player][infostate]

    def _get_average_policy(self, current_player, infostate):
        if infostate not in (player_policy := self.avg_policy[current_player]):
            player_policy[infostate] = defaultdict(float)
        return player_policy[infostate]

    def _get_information_state(self, current_player, state):
        infostate = state.information_state_string(current_player)
        if infostate not in (player_policy := self.curr_policy[current_player]):
            las = state.legal_actions()
            player_policy[infostate] = {action: 1 / len(las) for action in las}
        return infostate

    def _get_regret_table(self, current_player: int, infostate: str):
        if infostate not in self.regret_table[current_player]:
            self.regret_table[current_player][infostate] = defaultdict(float)

        return self.regret_table[current_player][infostate]

    def _action_value_dict(cls):
        return dict()

    def _apply_regret_matching(self):
        for player, player_policy in enumerate(self.curr_policy):
            for infostate, regret_dict in self.regret_table[player].items():
                regret_matching(player_policy[infostate], regret_dict)

    def iterate(
        self,
        updating_player: Optional[int] = None,
    ):
        if self._verbose:
            print(
                "\nIteration",
                self.iteration
                if self._simultaneous_updates
                else f"{self.iteration // 2} {(self.iteration % 2 + 1)}/2",
            )

        if updating_player is None and not self._simultaneous_updates:
            updating_player = self.iteration % 2

        root_reach_probabilities = {player: 1.0 for player in [-1] + self.n_players}

        self._traverse(
            self.root_state.clone(),
            root_reach_probabilities,
            updating_player,
        )
        self._apply_regret_matching()
        self.iteration += 1

    def _traverse(
        self,
        state: pyspiel.State,
        reach_prob: dict[Action, Probability],
        updating_player: Optional[int] = None,
    ):
        if state.is_terminal():
            return state.returns()

        action_values = self._action_value_dict()

        if state.is_chance_node():
            return self._traverse_chance_node(
                state, reach_prob, updating_player, action_values
            )
        else:
            curr_player = state.current_player()
            infostate = self._get_information_state(curr_player, state)
            state_value = self._traverse_player_node(
                state, infostate, reach_prob, updating_player, action_values
            )
            if self._simultaneous_updates or updating_player == curr_player:
                self._update_regret_and_avg_policy(
                    action_values, state_value, reach_prob, infostate, curr_player
                )
            return state_value

    def _traverse_chance_node(self, state, reach_prob, updating_player, action_values):
        state_value = np.zeros(len(self.n_players))
        for outcome, outcome_prob in state.chance_outcomes():
            next_state = state.child(outcome)

            child_reach_prob = deepcopy(reach_prob)
            child_reach_prob[state.current_player()] *= outcome_prob

            action_values[outcome] = self._traverse(
                next_state, child_reach_prob, updating_player
            )
            state_value += outcome_prob * np.asarray(action_values[outcome])
        return state_value

    def _traverse_player_node(
        self, state, infostate, reach_prob, updating_player, action_values
    ):
        curr_player = state.current_player()
        state_value = np.zeros(len(self.n_players))

        for action, action_prob in self._get_current_policy(
            curr_player, infostate
        ).items():
            child_reach_prob = deepcopy(reach_prob)
            child_reach_prob[curr_player] *= action_prob
            next_state = state.child(action)

            action_values[action] = self._traverse(
                next_state, child_reach_prob, updating_player
            )
            state_value += action_prob * np.asarray(action_values[action])
        return state_value

    def _update_regret_and_avg_policy(
        self, action_values, state_value, reach_prob, infostate, curr_player
    ):
        player_reach_prob = reach_prob[curr_player]
        cf_reach_prob = counterfactual_reach_prob(reach_prob, curr_player)
        # fetch the infostate specific policy tables for the current player
        avg_policy = self._get_average_policy(curr_player, infostate)
        curr_policy = self._get_current_policy(curr_player, infostate)
        regrets = self._get_regret_table(curr_player, infostate)
        for action, action_value in action_values.items():
            regrets[action] += cf_reach_prob * (
                action_value[curr_player] - state_value[curr_player]
            )
            avg_policy[action] += player_reach_prob * curr_policy[action]
