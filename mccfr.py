import warnings
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from functools import reduce
from typing import Optional, Dict
import numpy as np

import pyspiel
import torch.nn.functional

import rm
from rm import (
    regret_matching,
    counterfactual_reach_prob,
    kuhn_optimal_policy,
    KuhnAction,
    kuhn_poker_infostate_translation,
    Player,
)
from utils import timing

class MCCFRWeightingMode(Enum):
    lazy = 0
    optimistic = 1
    stochastic = 2


class FixedRng:
    def __init__(self):
        self._choices = [0, 1, 1, 1, 0, 1, 0, 1, 0]

    def choice(self, container, p):
        return container[self._choices.pop(0)]


class OutcomeSamplingMCCFR:
    def __init__(
        self,
        root_state: pyspiel.State,
        alternating_updates: bool,
        weighting_mode: MCCFRWeightingMode,
        curr_policy_list: list[Dict[str, Dict[int, float]]],
        average_policy_list: list[Dict[str, Dict[int, float]]],
        epsilon: float = 0.6,
        seed_or_rng: Optional[int] = None,
    ):
        self.root_state = root_state
        self.alternating_updates = alternating_updates
        self.weighting_mode = weighting_mode
        self.regret_table: list[Dict[str, Dict[int, float]]] = [
            {} for _ in range(root_state.num_players())
        ]
        self.curr_policy = curr_policy_list
        self.avg_policy = average_policy_list
        self.iteration = 0
        self.last_visit: defaultdict[str, int] = defaultdict(int)
        self.epsilon = epsilon
        self.rng: np.random.Generator
        if seed_or_rng is None or isinstance(seed_or_rng, int):
            self.rng = np.random.default_rng(seed_or_rng)
        else:
            self.rng = seed_or_rng

    def iterate(
        self,
        updating_player: Optional[int] = None,
        only_find_average_policy_value: bool = False,
    ):
        print("\nIteration number:", self.iteration)
        value, tail_prob = self._mccfr(
            deepcopy(self.root_state),
            {player: 1.0 for player in range(-1, self.root_state.num_players())},
            updating_player,
            sample_probability=1.0,
        )
        self.iteration += 1
        return value

    def _mccfr(
        self,
        state: pyspiel.State,
        reach_prob: dict[int, float],
        updating_player: Optional[int] = None,
        sample_probability=1.0,
    ):

        current_player = state.current_player()
        if state.is_terminal():
            reward, sample_prob = (
                state.player_return(updating_player) / sample_probability,
                1.0,
            )
            return reward, sample_prob

        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            aidx = self.rng.choice(list(range(len(outcomes))), p=probs)

            reach_prob[current_player] *= probs[aidx]
            state.apply_action(int(outcomes[aidx]))

            return self._mccfr(
                state, reach_prob, updating_player, sample_probability * probs[aidx],
            )

        infostate = state.information_state_string(current_player)

        legal_actions = state.legal_actions()
        player_policy = self._prefill_policies_at_infostate(
            current_player, infostate, legal_actions
        )

        regret_table = self.regret_table[current_player]
        if infostate not in regret_table:
            regret_table[infostate] = {action: 0.0 for action in legal_actions}

        regret_matching(player_policy, regret_table[infostate])
        (
            sampled_action,
            sampled_action_prob,
            sampled_action_sample_prob,
        ) = self._sample_action(current_player, updating_player, player_policy)

        child_reach_prob = deepcopy(reach_prob)
        child_reach_prob[current_player] *= sampled_action_prob

        state.apply_action(sampled_action)
        action_value, tail_prob = self._mccfr(
            state,
            child_reach_prob,
            updating_player,
            sample_probability * sampled_action_sample_prob,
        )
        print("Action Value", action_value)
        print("Action Value times sampled prob", action_value * sampled_action_prob)
        if not self.alternating_updates or updating_player == current_player:
            cf_value_weight = action_value * counterfactual_reach_prob(
                reach_prob, current_player
            )
            assert player_policy[sampled_action] == sampled_action_prob
            for action in regret_table[infostate].keys():
                if action == sampled_action:
                    incr = (
                        cf_value_weight
                        * tail_prob
                        * (1.0 - player_policy[sampled_action])
                    )

                    print("Action regret incr", incr)
                    regret_table[infostate][action] += incr
                else:
                    incr = -cf_value_weight * tail_prob * player_policy[sampled_action]

                    print("Action regret incr", incr)
                    regret_table[infostate][action] += incr
        else:
            avg_policy = self.avg_policy[current_player][infostate]

            if self.weighting_mode == MCCFRWeightingMode.optimistic:
                last_visit_difference = self.iteration + 1 - self.last_visit[infostate]
                for action in avg_policy.keys():
                    incr = (
                        reach_prob[current_player]
                        * player_policy[action]
                        * last_visit_difference
                    )
                    avg_policy[action] += incr
                self.last_visit[infostate] = self.iteration

            elif self.weighting_mode == MCCFRWeightingMode.stochastic:
                for action in avg_policy.keys():
                    incr = (
                        reach_prob[current_player]
                        * player_policy[action]
                        / sample_probability
                    )
                    avg_policy[action] += incr
        return action_value, tail_prob * sampled_action_prob

    def _prefill_policies_at_infostate(self, current_player, infostate, legal_actions):
        curr_player_policy = self.curr_policy[current_player]
        if infostate not in curr_player_policy:
            self.curr_policy[current_player][infostate] = {
                action: 1 / len(legal_actions) for action in legal_actions
            }
        if infostate not in self.avg_policy[current_player]:
            self.avg_policy[current_player][infostate] = {
                action: 0.0 for action in legal_actions
            }
        return curr_player_policy[infostate]

    def _sample_action(self, current_player, updating_player, policy):
        n_choices = len(policy)
        uniform_prob = 1.0 / n_choices
        sample_policy = {}
        normal_policy = {}
        choices = []
        if current_player == updating_player:
            for action, policy_prob in sorted(
                policy.items(), key=lambda pair: int(pair[0])
            ):
                choices.append(action)
                sample_policy[action] = self.epsilon * uniform_prob + (1 - self.epsilon) * policy_prob

                normal_policy[action] = policy_prob
        else:
            for action, policy_prob in sorted(
                policy.items(), key=lambda pair: int(pair[0])
            ):
                choices.append(action)
                sample_policy[action] = policy_prob
                normal_policy[action] = policy_prob

        sampled_action = self.rng.choice(choices, p=list(sample_policy.values()))
        sampled_action_prob = policy[sampled_action]
        sampled_action_sample_prob = sample_policy[sampled_action]
        return sampled_action, sampled_action_prob, sampled_action_sample_prob


class ExternalSamplingMCCFR:
    def __init__(
        self,
        root_state: pyspiel.State,
        curr_policy_list: list[Dict[str, Dict[int, float]]],
        average_policy_list: list[Dict[str, Dict[int, float]]],
        epsilon: float = 0.6,
        seed: Optional[int] = None,
    ):
        self.root_state = root_state
        self.regret_table: list[Dict[str, Dict[int, float]]] = [
            {} for p in range(root_state.num_players())
        ]
        self.curr_policy = curr_policy_list
        self.avg_policy = average_policy_list
        self.iteration = 0
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def iterate(self, updating_player: Optional[int] = None):
        print("\nIteration number:", self.iteration, "\n")
        value = self._mccfr(deepcopy(self.root_state), updating_player)
        self.iteration += 1
        return value

    def _mccfr(self, state: pyspiel.State, updating_player: int = 0):
        current_player = state.current_player()
        if state.is_terminal():
            reward = state.player_return(updating_player)
            return reward

        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            outcome, outcome_prob = self.rng.choice(outcomes)
            state.apply_action(int(outcome))

            return self._mccfr(state, updating_player)

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

                action_values[action] = self._mccfr(next_state, updating_player)

                state_value += player_policy[infostate][action] * action_values[action]
            curr_regret_table = regret_table[infostate]
            for action, regret in curr_regret_table.items():
                curr_regret_table[action] = regret + action_values[action] - state_value
            return state_value
        else:
            policy = player_policy[infostate]

            sample_policy = []
            choices = []
            for action, policy_prob in sorted(
                policy.items(), key=lambda pair: int(pair[0])
            ):
                choices.append(action)
                sample_policy.append(policy_prob)
            sampled_action = self.rng.choice(choices, p=sample_policy)
            # sampled_action = choices[0]
            state.apply_action(sampled_action)
            action_value = self._mccfr(state, updating_player)

            avg_policy = self.avg_policy[current_player][infostate]
            for action, prob in player_policy[infostate].items():
                avg_policy[action] += prob

            return action_value

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


def main(algorithm, n_iter, simultaneous_updates: bool = True, do_print: bool = True):

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
    solver = None
    if algorithm == "os":
        solver = OutcomeSamplingMCCFR(
            root_state,
            True,
            MCCFRWeightingMode.stochastic,
            current_policies,
            average_policies,
            seed_or_rng=0,
        )
    elif algorithm == "es":
        solver = ExternalSamplingMCCFR(
            root_state, current_policies, average_policies, seed=0
        )

    if do_print:
        print(
            f"Running Pure CFR with "
            f"{'simultaneous updates' if simultaneous_updates else 'alternating updates'} "
            f"for {n_iter} iterations."
        )

    solver = PureCFR(
        root_state,
        current_policies,
        average_policies,
        simultaneous_updates=simultaneous_updates,
        verbose=do_print,
    )
    for i in range(n_iter):
        solver.iterate()

        if sum(map(lambda p: len(p), solver.average_policy())) == len(
                all_infostates
        ) and (simultaneous_updates or (not simultaneous_updates and i > 1)):
            average_policy = solver.average_policy()
            expl_values.append(
                exploitability.exploitability(
                    game, to_pyspiel_tab_policy(average_policy),
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
    main("os",n_iter=20000, simultaneous_updates=True, do_print=True)
    # main(n_iter=20000, simultaneous_updates=False, do_print=True)
