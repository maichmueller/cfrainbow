from collections import defaultdict
from copy import deepcopy
from enum import Enum
from typing import Dict, Mapping, Optional

import numpy as np

import pyspiel

from rm import regret_matching, regret_matching_plus
from utils import (
    counterfactual_reach_prob
)
from type_aliases import Action, Infostate, Probability, Regret, Value

class Players(Enum):
    chance = -1
    player1 = 0
    player2 = 1


class DiscountedCFR:
    def __init__(
        self,
        root_state: pyspiel.State,
        curr_policy_list: list[Dict[Infostate, Mapping[Action, Probability]]],
        average_policy_list: list[Dict[Infostate, Mapping[Action, Probability]]],
        alpha: float = 3.0 / 2.0,
        beta: float = 0.0,
        gamma: float = 2,
        *,
        simultaneous_updates: bool = True,
        do_regret_matching_plus: bool = False,
        verbose: bool = False,
    ):
        self.root_state = root_state
        self.n_players = root_state.num_players()
        self.regret_table: list[Dict[Infostate, Dict[Action, Regret]]] = [
            {} for p in range(root_state.num_players())
        ]
        self.curr_policy = curr_policy_list
        self.avg_policy = average_policy_list
        self.iteration = 0
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._simultaneous_updates = simultaneous_updates
        self._do_rm_plus = do_regret_matching_plus
        self._verbose = verbose

    def _get_current_policy(self, current_player, infostate):
        return self.curr_policy[current_player][infostate]

    def _get_average_policy(self, current_player, infostate):
        if infostate not in self.avg_policy[current_player]:
            self.avg_policy[current_player][infostate] = defaultdict(float)
        return self.avg_policy[current_player][infostate]

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

    def _apply_weights(self, updating_player: Optional[int] = None):
        t = self.iteration + 1  # avoid iteration 0 weight
        alpha_weight = t**self.alpha
        alpha_weight /= alpha_weight + 1
        beta_weight = t**self.beta
        beta_weight /= beta_weight + 1
        regret_weights = (beta_weight, alpha_weight)
        gamma_weight = (t / (t + 1)) ** self.gamma

        for player, player_policy in (
            enumerate(self.curr_policy)
            if self._simultaneous_updates
            else [(updating_player, self.curr_policy[updating_player])]
        ):
            player_regret_table = self.regret_table[player]
            for infostate, regret_dict in player_regret_table.items():
                player_regret_table[infostate] = {
                    a: r * regret_weights[r > 0] for a, r in regret_dict.items()
                }

        for player, player_policy in (
            enumerate(self.avg_policy)
            if self._simultaneous_updates
            else [(updating_player, self.avg_policy[updating_player])]
        ):
            for infostate, policy in player_policy.items():
                player_policy[infostate] = {
                    a: s * gamma_weight for a, s in policy.items()
                }

    def _apply_regret_matching(self):
        for player, player_policy in enumerate(self.curr_policy):
            for infostate, regret_dict in self.regret_table[player].items():
                if self._do_rm_plus:
                    regret_matching_plus(player_policy[infostate], regret_dict)
                else:
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

        root_reach_probabilities = {player.value: 1.0 for player in Players}

        self._traverse(
            self.root_state.clone(),
            root_reach_probabilities,
            updating_player,
        )
        self._apply_weights(updating_player)
        self._apply_regret_matching()
        self.iteration += 1

    def _traverse(
        self,
        state: pyspiel.State,
        reach_prob: dict[Action, Probability],
        updating_player: Optional[int] = None,
    ):
        if state.is_terminal():
            reward = state.returns()
            return reward

        curr_player = state.current_player()
        action_values = {}
        state_value = np.zeros(self.n_players)

        if state.is_chance_node():
            for outcome, outcome_prob in state.chance_outcomes():
                next_state = state.child(outcome)

                child_reach_prob = deepcopy(reach_prob)
                child_reach_prob[state.current_player()] *= outcome_prob

                action_values[outcome] = self._traverse(
                    next_state, child_reach_prob, updating_player
                )
                state_value += outcome_prob * np.asarray(action_values[outcome])

            return state_value

        else:
            infostate = self._get_information_state(curr_player, state)

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

            if updating_player == curr_player or self._simultaneous_updates:
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

            return state_value

    def average_policy(self, player: Optional[int] = None):
        if player is None:
            return self.avg_policy
        else:
            return [self.avg_policy[player]]


class LinearCFR:
    def __init__(
        self,
        root_state: pyspiel.State,
        curr_policy_list: list[Dict[Infostate, Mapping[Action, Probability]]],
        average_policy_list: list[Dict[Infostate, Mapping[Action, Probability]]],
        *,
        simultaneous_updates: bool = True,
        do_regret_matching_plus: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            root_state,
            curr_policy_list,
            average_policy_list,
            simultaneous_updates=simultaneous_updates,
            do_regret_matching_plus=do_regret_matching_plus,
            verbose=verbose,
            alpha=1.0,
            beta=1.0,
            gamma=1.0,
        )
