import itertools
import warnings
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from functools import reduce
from typing import Dict, Mapping, Optional

from rm import kuhn_optimal_policy, regret_matching_plus, counterfactual_reach_prob
import pyspiel
import matplotlib.pyplot as plt

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


class CFRPlus:
    def __init__(
        self,
        root_state: pyspiel.State,
        curr_policy_list: list[Dict[Infostate, Mapping[Action, Probability]]],
        average_policy_list: list[Dict[Infostate, Mapping[Action, Probability]]],
        verbose: bool = False,
    ):
        self.root_state = root_state
        self.regret_table: list[Dict[Infostate, Dict[Action, Regret]]] = [
            {} for p in range(root_state.num_players())
        ]
        self.curr_policy = curr_policy_list
        self.avg_policy = average_policy_list
        self.iteration = 0
        self._verbose = verbose

    def _get_current_strategy(self, current_player, infostate):
        return self.curr_policy[current_player][infostate]

    def _get_average_strategy(self, current_player, infostate):
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

    def _apply_regret_matching(self):
        for player, player_policy in enumerate(self.curr_policy):
            for infostate, regret_dict in self.regret_table[player].items():
                regret_matching_plus(player_policy[infostate], regret_dict)

    def iterate(
        self, updating_player: Optional[int] = None,
    ):
        if self._verbose:
            print("\nIteration", self.iteration // 2, f"{(self.iteration % 2 + 1)}/{2}")

        if updating_player is None:
            updating_player = self.iteration % 2

        root_reach_probabilities = {player.value: 1.0 for player in Players}

        self._traverse(
            self.root_state.clone(), root_reach_probabilities, updating_player,
        )
        self._apply_regret_matching()
        self.iteration += 1

    def _traverse(
        self,
        state: pyspiel.State,
        reach_prob: dict[Action, Probability],
        updating_player: int,
    ):
        if state.is_terminal():
            reward = state.player_return(updating_player)
            return reward

        elif state.is_chance_node():
            action_values = {}
            state_value = 0.0

            for outcome, outcome_prob in state.chance_outcomes():
                next_state = state.child(outcome)

                child_reach_prob = deepcopy(reach_prob)
                child_reach_prob[state.current_player()] *= outcome_prob

                action_values[outcome] = self._traverse(
                    next_state, child_reach_prob, updating_player
                )
                state_value += outcome_prob * action_values[outcome]

            return state_value

        elif state.is_player_node():
            current_player = state.current_player()
            action_values = {}
            state_value = 0.0

            infostate = self._get_information_state(current_player, state)

            for action, action_prob in self._get_current_strategy(
                current_player, infostate
            ).items():
                child_reach_prob = deepcopy(reach_prob)
                child_reach_prob[current_player] *= action_prob
                next_state = state.child(action)

                action_values[action] = self._traverse(
                    next_state, child_reach_prob, updating_player
                )
                state_value += action_prob * action_values[action]

            if updating_player == current_player:
                player_reach_prob = reach_prob[current_player]
                cf_reach_prob = counterfactual_reach_prob(reach_prob, current_player)
                # fetch the infostate specific policy tables for the current player
                avg_policy = self._get_average_strategy(current_player, infostate)
                curr_policy = self._get_current_strategy(current_player, infostate)
                regrets = self._get_regret_table(current_player, infostate)

                for action, action_value in action_values.items():
                    regrets[action] += cf_reach_prob * (action_value - state_value)
                    avg_policy[action] += (
                        (self.iteration // 2 + 1)
                        * player_reach_prob
                        * curr_policy[action]
                    )

            return state_value

    def average_policy(self, player: Optional[int] = None):
        if player is None:
            return self.avg_policy
        else:
            return [self.avg_policy[player]]


def main(n_iter, do_print: bool = True):
    from open_spiel.python.algorithms import exploitability

    if do_print:
        print(f"Running CFR with alternating updates for {n_iter} iterations.")
    expl_values = []
    game = pyspiel.load_game("kuhn_poker")
    root_state = game.new_initial_state()
    n_players = list(range(root_state.num_players()))
    current_policies = [{} for _ in n_players]
    average_policies = [{} for _ in n_players]
    solver = CFRPlus(root_state, current_policies, average_policies, verbose=do_print)
    for i in range(n_iter):
        solver.iterate()

        expl_values.append(
            exploitability.exploitability(
                game, to_pyspiel_tab_policy(average_policies),
            )
        )

        if do_print:
            print(
                f"-------------------------------------------------------------"
                f"--> Exploitability {expl_values[-1]: .5f}"
            )
            print_policy_profile(deepcopy(average_policies))
            print(f"---------------------------------------------------------------")
    if do_print:
        print_final_policy_profile(average_policies)

    return expl_values


if __name__ == "__main__":
    main(n_iter=2000, do_print=False)
