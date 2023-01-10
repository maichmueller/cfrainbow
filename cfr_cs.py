import itertools
import warnings
from copy import deepcopy
from enum import Enum
from functools import reduce
from typing import Optional, Dict

from open_spiel.python.algorithms import exploitability

from rm import regret_matching, counterfactual_reach_prob, kuhn_optimal_policy
import pyspiel
from pyspiel import TabularPolicy
from pyspiel import *
import numpy as np

from utils import timing

class CFR:
    def __init__(
        self,
        root_state: pyspiel.State,
        alternating_updates: bool,
        curr_policy_list: list[Dict[str, Dict[int, float]]],
        average_policy_list: list[Dict[str, Dict[int, float]]],
    ):
        self.root_state = root_state
        self.alternating_updates = alternating_updates
        self.regret_table: list[Dict[str, Dict[int, float]]] = [
            {} for p in range(root_state.num_players())
        ]
        self.curr_policy = curr_policy_list
        self.avg_policy = average_policy_list
        self.iteration = 0
        self.rng = np.random.default_rng(0)

    def iterate(
        self,
        updating_player: Optional[int] = None,
        only_find_average_policy_value: bool = False,
    ):
        print("\nIteration", self.iteration, "\n")
        values = self._cfr(
            deepcopy(self.root_state),
            {player: 1.0 for player in range(-1, self.root_state.num_players())},
            updating_player,
            use_current_policy=not only_find_average_policy_value,
        )
        if not only_find_average_policy_value:
            self.apply_regret_matching()
        self.iteration += 1
        return values

    def _cfr(
        self,
        state: pyspiel.State,
        reach_prob: dict[int, float],
        updating_player: Optional[int] = None,
        use_current_policy: bool = True,
    ):
        if state.is_terminal():
            rewards = state.returns()
            return rewards

        current_player = state.current_player()
        state_value = [0.0] * state.num_players()
        action_values = {}

        if state.is_chance_node():
            outcome, outcome_prob = self.rng.choice(state.chance_outcomes())

            next_state = state.child(int(outcome))

            return self._cfr(
                next_state, reach_prob, updating_player, use_current_policy
            )

        player_policy = (
            self.curr_policy[current_player]
            if use_current_policy
            else self.avg_policy[current_player]
        )

        infostate = state.information_state_string(current_player)
        if use_current_policy and infostate not in player_policy:
            las = state.legal_actions()
            self.curr_policy[current_player][infostate] = {
                action: 1 / len(las) for action in las
            }
        for action, action_prob in player_policy[infostate].items():
            child_reach_prob = deepcopy(reach_prob)
            child_reach_prob[current_player] *= action_prob
            next_state = state.child(action)

            action_values[action] = self._cfr(
                next_state, child_reach_prob, updating_player, use_current_policy
            )

            state_value = [
                player_value + action_prob * child_value
                for player_value, child_value in zip(state_value, action_values[action])
            ]

        if use_current_policy:
            if self.alternating_updates:
                if updating_player == current_player:
                    self.update_regret_and_policy(
                        current_player,
                        infostate,
                        reach_prob,
                        state_value,
                        action_values,
                    )
            else:
                self.update_regret_and_policy(
                    current_player, infostate, reach_prob, state_value, action_values
                )
        return state_value

    def update_regret_and_policy(
        self,
        current_player: int,
        infostate: str,
        reach_prob: dict[int, float],
        state_value: list[float],
        action_values: dict[int, list[float]],
    ):
        player_reach_prob = reach_prob[current_player]
        cf_reach_prob = counterfactual_reach_prob(reach_prob, current_player)
        if infostate not in self.avg_policy[current_player]:
            avg_policy = {action: 0.0 for action in action_values.keys()}
            self.avg_policy[current_player][infostate] = avg_policy
        else:
            avg_policy = self.avg_policy[current_player][infostate]

        curr_policy = self.curr_policy[current_player][infostate]

        if infostate not in self.regret_table[current_player]:
            regrets = {action: 0.0 for action in action_values.keys()}
            self.regret_table[current_player][infostate] = regrets
        else:
            regrets = self.regret_table[current_player][infostate]

        for action, action_value in action_values.items():
            regrets[action] += cf_reach_prob * (
                action_value[current_player] - state_value[current_player]
            )
            avg_policy[action] += player_reach_prob * curr_policy[action]

    def apply_regret_matching(self):
        for player, player_policy in enumerate(self.curr_policy):
            for infostate, regret_dict in self.regret_table[player].items():
                regret_matching(player_policy[infostate], regret_dict)

@timing
def main():
    n_iterations = 200000
    # game = pyspiel.load_game("kuhn_poker")
    game = load_game_as_turn_based("matrix_rps")
    root_state = game.new_initial_state()
    print([root_state.action_to_string(action) for action in root_state.legal_actions()])
    # current_policies = [{} for player in range(root_state.num_players())]
    current_policies = [
        {
            f"Current player: {0}\nObserving player: {0}. Non-terminal": {
                0: 0.1,
                1: 0.2,
                2: 0.7,
            }
        },
        {
            f"Current player: {1}\nObserving player: {1}. Non-terminal": {
                0: 0.9,
                1: 0.05,
                2: 0.05,
            }
        }
    ]
    average_policies = [{} for player in range(root_state.num_players())]
    solver = CFR(root_state, False, current_policies, average_policies)
    for i in range(n_iterations):
        update_player = i % 2
        solver.iterate(update_player)
        if (len(average_policies[0]) + len(average_policies[1]) == 2) and (
            i + 1
        ) % 2 == 0:
            average_policies_copy = deepcopy(average_policies)
            for player_avg_policy in average_policies_copy:
                for infostate, avg_policy in player_avg_policy.items():
                    prob_sum = 0.0
                    for action, prob in avg_policy.items():
                        prob_sum += prob
                    if prob_sum > 0:
                        for action in avg_policy.keys():
                            avg_policy[action] /= prob_sum
            policy_profile = {
                istate: [
                    (action, prob) for action, prob in enumerate(as_and_ps.values())
                ]
                for istate, as_and_ps in itertools.chain(
                    average_policies_copy[0].items(), average_policies_copy[1].items()
                )
            }
            expl = exploitability(
                game,
                pyspiel.TabularPolicy(policy_profile),
                # use_cpp_br=True
            )
            print(expl)
            if expl < 1e-3:
                break
    alpha = average_policies[0]["0"][1] / sum(average_policies[0]["0"].values())
    if alpha > 1 / 3:
        warnings.warn(f"{alpha=} is greater than 1/3")
    else:
        print(f"{alpha=:.2f}")
    optimal_for_alpha = kuhn_optimal_policy(alpha)

    for i, player_avg_policy in enumerate(average_policies):
        for infostate, avg_policy in player_avg_policy.items():
            prob_sum = 0.0
            for action, prob in avg_policy.items():
                prob_sum += prob
            if prob_sum > 0:
                for action in avg_policy.keys():
                    avg_policy[action] /= prob_sum
        print("player", i)
        for infostate, dist in player_avg_policy.items():
            print(
                infostate,
                list(f"{action}: {round(prob, 2)}" for action, prob in dist.items()),
            )

    print("\ntheoretically optimal policy:\n")
    for i, player_avg_policy in optimal_for_alpha.items():
        print("player", i)
        for infostate, dist in player_avg_policy.items():
            print(
                infostate,
                list(f"{action}: {round(prob, 2)}" for action, prob in dist.items()),
            )

    print("\nDifference to theoretically optimal policy:\n")
    for i, player_avg_policy in enumerate(average_policies):
        for infostate, avg_policy in player_avg_policy.items():
            prob_sum = 0.0
            for action, prob in avg_policy.items():
                prob_sum += prob
            if prob_sum > 0:
                for action in avg_policy.keys():
                    avg_policy[action] /= prob_sum
        print("player", i)
        for infostate, dist in player_avg_policy.items():
            print(
                infostate,
                list(
                    f"{action}: {round(prob - optimal_for_alpha[i][infostate][action], 2)}"
                    for action, prob in dist.items()
                ),
            )

    print(abs(solver.iterate(None, True)[0] + 1 / 18))


if __name__ == "__main__":
    main()
