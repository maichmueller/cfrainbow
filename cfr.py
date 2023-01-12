from collections import defaultdict
from copy import deepcopy
from enum import Enum
from typing import Dict, Mapping, Optional

import numpy as np

from rm import counterfactual_reach_prob, regret_matching
import pyspiel
from open_spiel.python.algorithms import exploitability
from utils import (
    print_final_policy_profile,
    print_policy_profile,
    to_pyspiel_tab_policy,
)


Action = int
Probability = float
Regret = float
Value = float
Infostate = str


class Players(Enum):
    chance = -1
    player1 = 0
    player2 = 1


class CFR:
    def __init__(
        self,
        root_state: pyspiel.State,
        curr_policy_list: list[Dict[Infostate, Mapping[Action, Probability]]],
        average_policy_list: list[Dict[Infostate, Mapping[Action, Probability]]],
        simultaneous_updates: bool = True,
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
        self._simultaneous_updates = simultaneous_updates
        self._verbose = verbose

    def _get_current_strategy(self, current_player, infostate):
        return self.curr_policy[current_player][infostate]

    def _get_average_strategy(self, current_player, infostate):
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

        root_reach_probabilities = {player.value: 1.0 for player in Players}

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
            reward = state.returns()
            return reward

        curr_player = state.current_player()
        action_values = {}

        if state.is_chance_node():
            return self._traverse_chance_node(
                state, reach_prob, updating_player, action_values
            )
        else:
            infostate = self._get_information_state(curr_player, state)
            state_value = self._traverse_player_node(
                state, infostate, reach_prob, updating_player, action_values
            )
            if self._simultaneous_updates or updating_player == curr_player:
                self._update_regret_and_avg_strategy(
                    action_values, state_value, reach_prob, infostate, curr_player
                )
            return state_value

    def _update_regret_and_avg_strategy(
        self, action_values, state_value, reach_prob, infostate, curr_player
    ):
        player_reach_prob = reach_prob[curr_player]
        cf_reach_prob = counterfactual_reach_prob(reach_prob, curr_player)
        # fetch the infostate specific policy tables for the current player
        avg_policy = self._get_average_strategy(curr_player, infostate)
        curr_policy = self._get_current_strategy(curr_player, infostate)
        regrets = self._get_regret_table(curr_player, infostate)
        for action, action_value in action_values.items():
            regrets[action] += cf_reach_prob * (
                action_value[curr_player] - state_value[curr_player]
            )
            avg_policy[action] += player_reach_prob * curr_policy[action]

    def _traverse_chance_node(self, state, reach_prob, updating_player, action_values):
        state_value = np.zeros(self.n_players)
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
        state_value = np.zeros(self.n_players)

        for action, action_prob in self._get_current_strategy(
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

    def average_policy(self, player: Optional[int] = None):
        if player is None:
            return self.avg_policy
        else:
            return [self.avg_policy[player]]


def main(n_iter, simultaneous_updates: bool = True, do_print: bool = True):
    if do_print:
        print(
            f"Running CFR with {'simultaneous updates' if simultaneous_updates else 'alternating updates'} for {n_iter} iterations."
        )

    expl_values = []
    game = pyspiel.load_game("kuhn_poker")
    root_state = game.new_initial_state()
    n_players = list(range(root_state.num_players()))
    current_policies = [{} for _ in n_players]
    average_policies = [{} for _ in n_players]
    solver = CFR(
        root_state,
        current_policies,
        average_policies,
        simultaneous_updates=simultaneous_updates,
        verbose=do_print,
    )
    for i in range(n_iter):
        solver.iterate()

        if simultaneous_updates or (not simultaneous_updates and i > 1):
            expl_values.append(
                exploitability.exploitability(
                    game,
                    to_pyspiel_tab_policy(average_policies),
                )
            )

            if do_print:
                print(
                    f"-------------------------------------------------------------"
                    f"--> Exploitability {expl_values[-1]: .5f}"
                )
                print_policy_profile(deepcopy(average_policies))
                print(
                    f"---------------------------------------------------------------"
                )
    if do_print:
        print_final_policy_profile(average_policies)

    return expl_values


if __name__ == "__main__":
    main(n_iter=2000, do_print=True)
