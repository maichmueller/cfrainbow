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


class MCCFRWeightingMode(Enum):
    lazy = 0
    optimistic = 1
    stochastic = 2


class OutcomeSamplingMCCFR:
    def __init__(
        self,
        root_state: pyspiel.State,
        curr_policy_list: list[Dict[str, Dict[int, float]]],
        average_policy_list: list[Dict[str, Dict[int, float]]],
        *,
        weighting_mode: MCCFRWeightingMode,
        epsilon: float = 0.6,
        simultaneous_updates: bool,
        verbose: bool = False,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        self.root_state = root_state
        self.n_players = list(range(root_state.num_players()))
        self.weighting_mode = weighting_mode
        self.regret_table: list[Dict[str, Dict[int, float]]] = [
            {} for _ in self.n_players
        ]
        self.curr_policy = curr_policy_list
        self.avg_policy = average_policy_list
        self.iteration = 0
        self.last_visit: defaultdict[str, int] = defaultdict(int)
        self.weight_storage: Dict[str, Dict[int, float]] = dict()
        self.epsilon = epsilon
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self._simultaneous_updates = simultaneous_updates
        self._verbose = verbose

    def average_policy(self):
        return self.avg_policy

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
        if infostate not in (table := self.regret_table[current_player]):
            table[infostate] = defaultdict(float)
        return table[infostate]

    def _get_weight_storage(self, infostate):
        if infostate not in self.weight_storage:
            self.weight_storage[infostate] = defaultdict(float)
        return self.weight_storage[infostate]

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

        value, tail_prob = self._mccfr(
            deepcopy(self.root_state),
            {player: 1.0 for player in [-1] + self.n_players},
            updating_player,
            sample_probability=1.0,
            weights={player: 0.0 for player in self.n_players}
            if self.weighting_mode == MCCFRWeightingMode.lazy
            else None,
        )
        self.iteration += 1
        return value

    def _mccfr(
        self,
        state: pyspiel.State,
        reach_prob: dict[int, float],
        updating_player: Optional[int] = None,
        sample_probability=1.0,
        weights: Optional[dict[int, float]] = None,
    ):

        curr_player = state.current_player()
        if state.is_terminal():
            reward, sample_prob = (
                np.asarray(state.returns()) / sample_probability,
                1.0,
            )
            return reward, sample_prob

        if state.is_chance_node():
            chance_policy = {a: p for a, p in state.chance_outcomes()}

            sampled_action = self.rng.choice(
                list(chance_policy.keys()), p=list(chance_policy.values())
            )
            reach_prob[curr_player] *= chance_policy[sampled_action]
            state.apply_action(int(sampled_action))

            return self._mccfr(
                state,
                reach_prob,
                updating_player,
                sample_probability * chance_policy[sampled_action],
                weights=weights,
            )

        infostate = self._get_information_state(curr_player, state)
        player_policy = self._get_current_strategy(curr_player, infostate)
        regret_table = self._get_regret_table(curr_player, infostate)

        regret_matching(player_policy, regret_table)

        (
            sampled_action,
            sampled_action_prob,
            sampled_action_sample_prob,
        ) = self._sample_action(curr_player, updating_player, player_policy)

        child_reach_prob = deepcopy(reach_prob)
        child_reach_prob[curr_player] *= sampled_action_prob
        next_weights = deepcopy(weights)
        if self.weighting_mode == MCCFRWeightingMode.lazy:
            next_weights[curr_player] = (
                next_weights[curr_player] * sampled_action_prob
                + self._get_weight_storage(infostate)[sampled_action]
            )

        state.apply_action(sampled_action)
        action_value, tail_prob = self._mccfr(
            state,
            child_reach_prob,
            updating_player,
            sample_probability * sampled_action_sample_prob,
            weights=next_weights,
        )
        if self._simultaneous_updates or updating_player == curr_player:
            cf_value_weight = action_value[curr_player] * counterfactual_reach_prob(
                reach_prob, curr_player
            )
            for action in player_policy.keys():
                is_sampled_action = action == sampled_action
                regret_table[action] += (
                    cf_value_weight
                    * tail_prob
                    * (
                        is_sampled_action * (1.0 - player_policy[sampled_action])
                        - (not is_sampled_action) * player_policy[sampled_action]
                    )
                )
            if self._simultaneous_updates:
                self._update_average_strategy(
                    curr_player,
                    infostate,
                    player_policy,
                    sampled_action,
                    reach_prob,
                    sample_probability,
                    weights,
                )
        else:
            self._update_average_strategy(
                curr_player,
                infostate,
                player_policy,
                sampled_action,
                reach_prob,
                sample_probability,
                weights,
            )
        return action_value, tail_prob * sampled_action_prob

    def _update_average_strategy(
        self,
        curr_player,
        infostate,
        player_policy,
        sampled_action,
        reach_prob,
        sample_probability,
        weights,
    ):
        avg_policy = self._get_average_strategy(curr_player, infostate)
        if self.weighting_mode == MCCFRWeightingMode.optimistic:
            last_visit_difference = self.iteration + 1 - self.last_visit[infostate]
            self.last_visit[infostate] = self.iteration
            for action, policy_prob in player_policy.items():
                avg_policy[action] += (
                    reach_prob[curr_player] * policy_prob * last_visit_difference
                )
        elif self.weighting_mode == MCCFRWeightingMode.stochastic:
            for action, policy_prob in player_policy.items():
                avg_policy[action] += (
                    reach_prob[curr_player] * policy_prob / sample_probability
                )
        else:
            # lazy weighting updates
            for action, policy_prob in player_policy.items():
                policy_incr = (
                    weights[curr_player] + reach_prob[curr_player]
                ) * policy_prob
                avg_policy[action] += policy_incr
                weights[curr_player] = (weights[curr_player] + policy_incr) * (
                    action != sampled_action
                )

    def _sample_action(self, current_player, updating_player, policy):
        n_choices = len(policy)
        uniform_prob = 1.0 / n_choices
        sample_policy = {}
        normal_policy = {}
        choices = []

        epsilon = self.epsilon if current_player == updating_player else 0.0

        for action, policy_prob in policy.items():
            choices.append(action)
            sample_policy[action] = epsilon * uniform_prob + (1 - epsilon) * policy_prob
            normal_policy[action] = policy_prob

        sampled_action = self.rng.choice(choices, p=list(sample_policy.values()))
        return sampled_action, policy[sampled_action], sample_policy[sampled_action]


def main(
    n_iter,
    simultaneous_updates: bool = True,
    weighting_mode: Union[int, MCCFRWeightingMode] = 2,
    do_print: bool = True,
):

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

    solver = OutcomeSamplingMCCFR(
        root_state,
        current_policies,
        average_policies,
        weighting_mode=MCCFRWeightingMode(weighting_mode),
        simultaneous_updates=simultaneous_updates,
        verbose=do_print,
        seed=0
    )

    if do_print:
        print(
            f"Running Outcome Sampling MCCFR with "
            f"{'simultaneous updates' if simultaneous_updates else 'alternating updates'} and "
            f"{MCCFRWeightingMode(weighting_mode).name} weighting"
            f"for {n_iter} iterations."
        )

    for i in range(n_iter):
        solver.iterate()

        if sum(map(lambda p: len(p), solver.average_policy())) == len(
            all_infostates
        ) and (simultaneous_updates or (not simultaneous_updates and i > 1)):
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
    main(n_iter=200000, simultaneous_updates=False, weighting_mode=0, do_print=True)
