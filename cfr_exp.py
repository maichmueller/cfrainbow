from collections import defaultdict
from copy import deepcopy
from enum import Enum
from typing import Dict, Mapping, Optional

import numpy as np

from rm import counterfactual_reach_prob, regret_matching
import pyspiel

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


class Player(Enum):
    chance = -1
    player1 = 0
    player2 = 1


class ExponentialCFR:
    def __init__(
        self,
        root_state: pyspiel.State,
        curr_policy_list: list[Dict[Infostate, Mapping[Action, Probability]]],
        average_policy_list: list[Dict[Infostate, Mapping[Action, Probability]]],
        simultaneous_updates: bool = True,
        verbose: bool = False,
    ):
        self.root_state = root_state
        self.n_players = list(range(root_state.num_players()))
        self.regret_table: list[Dict[Infostate, Dict[Action, Regret]]] = [
            {} for p in self.n_players
        ]
        self.curr_policy = curr_policy_list
        self.avg_policy_numerator = average_policy_list
        self.avg_policy_denominator: tuple[
            Dict[Infostate, Dict[Action, Probability]]
        ] = tuple(defaultdict(dict) for _ in self.n_players)
        self.iteration = 0

        self._regret_increments: tuple[Dict[Infostate, Dict[Action, Regret]]] = tuple(
            defaultdict(dict) for _ in self.n_players
        )
        self._reach_prob: Dict[Infostate, Probability] = {}
        self._simultaneous_updates = simultaneous_updates
        self._verbose = verbose

    def _get_current_strategy(self, player: int, infostate: Infostate):
        return self.curr_policy[player][infostate]

    def _get_avg_policy(
        self, player: int, infostate: Infostate, numerator: bool = True
    ):
        if infostate not in (
            regret_table := self.avg_policy_numerator[player]
            if numerator
            else self.avg_policy_denominator[player]
        ):
            regret_table[infostate] = defaultdict(float)
        return regret_table[infostate]

    def _get_information_state(self, player: int, state: pyspiel.State):
        infostate = state.information_state_string(player)
        if infostate not in (player_policy := self.curr_policy[player]):
            las = state.legal_actions()
            player_policy[infostate] = {action: 1 / len(las) for action in las}
        return infostate

    def _get_regret_table(self, player: int, infostate: Infostate, temp: bool = False):
        if temp:
            regret_table = self._regret_increments[player]
        else:
            regret_table = self.regret_table[player]

        if infostate not in regret_table:
            regret_table[infostate] = defaultdict(float)
        return regret_table[infostate]

    def _apply_regret_matching(self):
        for player, player_policy in enumerate(self.curr_policy):
            for infostate, regret_dict in self.regret_table[player].items():
                regret_matching(player_policy[infostate], regret_dict)

    def average_policy(self, player: Optional[int] = None):
        policy_out = []
        for player in self.n_players if player is None else (player,):
            policy_out.append(defaultdict(dict))
            player_numerator_table = self.avg_policy_numerator[player]
            player_denominator_table = self.avg_policy_denominator[player]
            for infostate, policy in player_numerator_table.items():
                policy_denom = player_denominator_table[infostate]
                for action, prob in policy.items():
                    policy_out[player][infostate][action] = prob / policy_denom[action]
        return policy_out

    def iterate(
        self, updating_player: Optional[int] = None,
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

        root_reach_probabilities = {player.value: 1.0 for player in Player}

        self._traverse(
            self.root_state.clone(), root_reach_probabilities, updating_player,
        )
        self._apply_exponential_weight(updating_player)
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
        state_value = np.zeros(len(self.n_players))

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

            if updating_player == curr_player or self._simultaneous_updates:
                cf_reach_prob = counterfactual_reach_prob(reach_prob, curr_player)
                # fetch the infostate specific policy tables for the current player
                regrets = self._get_regret_table(curr_player, infostate, temp=True)
                # set the player reach prob for this infostate
                # (the probability of only the owning player playing to this infostate)
                self._reach_prob[infostate] = reach_prob[curr_player]

                for action, action_value in action_values.items():
                    regrets[action] += cf_reach_prob * (
                        action_value[curr_player] - state_value[curr_player]
                    )

            return state_value

    def _apply_exponential_weight(self, updating_player: Optional[int] = None):
        for player, player_regret_incrs in (
            enumerate(self._regret_increments)
            if self._simultaneous_updates
            else [(updating_player, self._regret_increments[updating_player])]
        ):
            for infostate, regret_incrs in player_regret_incrs.items():
                avg_regret = sum(regret_incrs.values()) / len(regret_incrs)
                cumulative_regret_table = self._get_regret_table(
                    player, infostate, temp=False
                )
                strategy_numerator = self._get_avg_policy(
                    player, infostate, numerator=True
                )
                strategy_denominator = self._get_avg_policy(
                    player, infostate, numerator=False
                )
                reach_prob = self._reach_prob[infostate]

                for action, regret in regret_incrs.items():
                    exp_l1 = np.exp(regret - avg_regret)
                    policy_weight = exp_l1 * reach_prob

                    cumulative_regret_table[action] += exp_l1 * regret
                    strategy_numerator[action] += (
                        policy_weight
                        * self._get_current_strategy(player, infostate)[action]
                    )
                    strategy_denominator[action] += policy_weight
                # delete the content of the temporary regret incr table
                regret_incrs.clear()


def main(n_iter, simultaneous_updates: bool = True, do_print: bool = True):
    from open_spiel.python.algorithms import exploitability

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
    solver = ExponentialCFR(
        root_state,
        current_policies,
        average_policies,
        simultaneous_updates=simultaneous_updates,
        verbose=do_print,
    )
    for i in range(n_iter):
        solver.iterate()

        if simultaneous_updates or (not simultaneous_updates and i > 1):
            avg_policy = solver.average_policy()
            expl_values.append(
                exploitability.exploitability(game, to_pyspiel_tab_policy(avg_policy),)
            )

            if do_print:
                print(
                    f"-------------------------------------------------------------"
                    f"--> Exploitability {expl_values[-1]: .5f}"
                )
                print_policy_profile(avg_policy)
                print(
                    f"---------------------------------------------------------------"
                )
    if do_print:
        print_final_policy_profile(solver.average_policy())

    return expl_values


if __name__ == "__main__":
    main(n_iter=2000, do_print=True)
