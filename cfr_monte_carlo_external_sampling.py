from collections import defaultdict
from copy import deepcopy
from enum import Enum
from typing import Optional, Dict, Union
import numpy as np

import pyspiel
from open_spiel.python.algorithms import exploitability

import rm
from rm import regret_matching, counterfactual_reach_prob
from utils import (
    to_pyspiel_tab_policy,
    print_policy_profile,
    print_final_policy_profile,
)


class ExternalSamplingMCCFR:
    def __init__(
        self,
        root_state: pyspiel.State,
        curr_policy_list: list[Dict[str, Dict[int, float]]],
        average_policy_list: list[Dict[str, Dict[int, float]]],
        *,
        verbose: bool = False,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        self.root_state = root_state
        self.regret_table: list[Dict[str, Dict[int, float]]] = [
            {} for p in range(root_state.num_players())
        ]
        self.curr_policy = curr_policy_list
        self.avg_policy = average_policy_list
        self.iteration = 0
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self._verbose = verbose

    def average_policy(self):
        return self.avg_policy

    def iterate(self, updating_player: Optional[int] = None):

        if self._verbose:
            print(
                "\nIteration",
                f"{self.iteration // 2} {(self.iteration % 2 + 1)}/2",
            )

        if updating_player is None:
            updating_player = self.iteration % 2

        value = self._traverse(deepcopy(self.root_state), updating_player)
        self.iteration += 1
        return value

    def _traverse(self, state: pyspiel.State, updating_player: int = 0):
        current_player = state.current_player()
        if state.is_terminal():
            reward = state.player_return(updating_player)
            return reward

        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            outcome, outcome_prob = self.rng.choice(outcomes)
            state.apply_action(int(outcome))

            return self._traverse(state, updating_player)

        infostate = state.information_state_string(current_player)

        player_policy = self.curr_policy[current_player]

        self._prefill_policy_tables(current_player, state, infostate)

        regret_table = self.regret_table[current_player]
        if infostate not in regret_table:
            regret_table[infostate] = {action: 0.0 for action in state.legal_actions()}

        regret_matching(player_policy[infostate], regret_table[infostate])

        if updating_player == current_player:
            state_value = 0.0
            action_values = dict()

            for action in player_policy[infostate].keys():
                next_state = state.child(action)

                action_values[action] = self._traverse(next_state, updating_player)

                state_value += player_policy[infostate][action] * action_values[action]
            curr_regret_table = regret_table[infostate]
            for action, regret in curr_regret_table.items():
                curr_regret_table[action] = regret + action_values[action] - state_value
            return state_value
        else:
            policy = player_policy[infostate]

            sampled_action = self._sample_action(policy)

            state.apply_action(sampled_action)
            action_value = self._traverse(state, updating_player)

            avg_policy = self.avg_policy[current_player][infostate]
            for action, prob in player_policy[infostate].items():
                avg_policy[action] += prob

            return action_value

    def _sample_action(self, policy):
        sample_policy = []
        choices = []
        for action, policy_prob in policy.items():
            choices.append(action)
            sample_policy.append(policy_prob)
        sampled_action = self.rng.choice(choices, p=sample_policy)
        return sampled_action

    def _prefill_policy_tables(self, current_player, state, infostate):
        if infostate not in self.curr_policy[current_player]:
            las = state.legal_actions()
            self.curr_policy[current_player][infostate] = {
                action: 1 / len(las) for action in state.legal_actions()
            }
        if infostate not in self.avg_policy[current_player]:
            self.avg_policy[current_player][infostate] = {
                action: 0.0 for action in state.legal_actions()
            }


def main(n_iter, do_print: bool = True):

    expl_values = []
    game = pyspiel.load_game("kuhn_poker")
    root_state = game.new_initial_state()
    n_players = list(range(root_state.num_players()))
    current_policies = [{} for _ in n_players]
    average_policies = [{} for _ in n_players]
    all_infostates = {
        state.information_state_string(state.current_player())
        for state in rm.all_states_gen(game=game)
    }

    if do_print:
        print(
            f"Running External Sampling MCCFR "
            f"for {n_iter} iterations."
        )

    solver = ExternalSamplingMCCFR(
        root_state,
        current_policies,
        average_policies,
        verbose=do_print,
        seed=0,
    )
    for i in range(n_iter):
        solver.iterate()

        if sum(map(lambda p: len(p), solver.average_policy())) == len(
            all_infostates
        ):
            average_policy = solver.average_policy()
            expl_values.append(
                exploitability.exploitability(
                    game,
                    to_pyspiel_tab_policy(average_policy),
                )
            )

            if do_print:
                print(
                    f"-------------------------------------------------------------"
                    f"--> Exploitability {expl_values[-1]: .5f}"
                )
                print_policy_profile(deepcopy(average_policy))
                print(
                    f"---------------------------------------------------------------"
                )
    if do_print:
        print_final_policy_profile(solver.average_policy())

    return expl_values


if __name__ == "__main__":
    main(n_iter=20000, do_print=True)
