from collections import defaultdict
from copy import deepcopy
from enum import Enum
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from cfr_base import CFRBase
from rm import regret_matching
import pyspiel

from utils import counterfactual_reach_prob
from type_aliases import Action, Infostate, Probability, Regret, Value


class ExponentialCFR(CFRBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.avg_policy_denominator: Sequence[
            Dict[Infostate, Dict[Action, Probability]]
        ] = [dict() for _ in self.players]
        self._regret_increments: Sequence[Dict[Infostate, Dict[Action, Regret]]] = [
            dict() for _ in self.players
        ]
        self._reach_prob: Dict[Infostate, Probability] = {}

    def _avg_policy_at(self, player: int, infostate: Infostate, numerator: bool = True):
        if infostate not in (
            policy_table := self._avg_policy[player]
            if numerator
            else self.avg_policy_denominator[player]
        ):
            policy_table[infostate] = {a: 0.0 for a in self.action_list(infostate)}
        return policy_table[infostate]

    def average_policy(self, player: Optional[int] = None):
        policy_out = []
        for player in self.players if player is None else (player,):
            policy_out.append(defaultdict(dict))
            player_numerator_table = self._avg_policy[player]
            player_denominator_table = self.avg_policy_denominator[player]
            for infostate, policy in player_numerator_table.items():
                policy_denom = player_denominator_table[infostate]
                for action, prob in policy.items():
                    policy_out[player][infostate][action] = prob / policy_denom[action]
        return policy_out

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

        root_reach_probabilities = {player: 1.0 for player in [-1] + self.players}

        self._traverse(
            self.root_state.clone(),
            root_reach_probabilities,
            traversing_player,
        )
        self._apply_exponential_weight(traversing_player)
        self._iteration += 1

    def _traverse(
        self,
        state: pyspiel.State,
        reach_prob: dict[Action, Probability],
        updating_player: Optional[int] = None,
    ):
        if state.is_terminal():
            reward = state.returns()
            return reward

        action_values = {}
        state_value = np.zeros(self.nr_players)

        if state.is_chance_node():
            return self._traverse_chance_node(
                state, reach_prob, updating_player, action_values, state_value
            )

        else:
            curr_player = state.current_player()
            infostate = state.information_state_string(curr_player)
            self._set_action_list(infostate, state)

            state_value = self._traverse_player_node(
                state,
                reach_prob,
                updating_player,
                infostate,
                curr_player,
                action_values,
                state_value,
            )

            if self.simultaneous or updating_player == curr_player:
                cf_reach_prob = counterfactual_reach_prob(reach_prob, curr_player)
                regrets = self._regret_increments_of(curr_player, infostate)
                for action, action_value in action_values.items():
                    regrets[action] += cf_reach_prob * (
                        action_value[curr_player] - state_value[curr_player]
                    )

                # set the player reach prob for this infostate
                # (the probability of only the owning player playing to this infostate)
                self._reach_prob[infostate] = reach_prob[curr_player]

            return state_value

    def _regret_increments_of(self, curr_player, infostate):
        if infostate not in (regret_table := self._regret_increments[curr_player]):
            regret_table[infostate] = {a: 0.0 for a in self.action_list(infostate)}
        regrets = regret_table[infostate]
        return regrets

    def _traverse_player_node(
        self,
        state,
        reach_prob,
        updating_player,
        infostate,
        curr_player,
        action_values,
        state_value,
    ):
        player_policy = self.regret_minimizer(infostate).recommend(self.iteration)
        for action, action_prob in player_policy.items():
            child_reach_prob = deepcopy(reach_prob)
            child_reach_prob[curr_player] *= action_prob

            action_values[action] = self._traverse(
                state.child(action), child_reach_prob, updating_player
            )
            state_value += action_prob * np.asarray(action_values[action])
        return state_value

    def _traverse_chance_node(
        self, state, reach_prob, updating_player, action_values, state_value
    ):
        for outcome, outcome_prob in state.chance_outcomes():
            next_state = state.child(outcome)

            child_reach_prob = deepcopy(reach_prob)
            child_reach_prob[state.current_player()] *= outcome_prob

            action_values[outcome] = self._traverse(
                next_state, child_reach_prob, updating_player
            )
            state_value += outcome_prob * np.asarray(action_values[outcome])
        return state_value

    def _apply_exponential_weight(self, updating_player: Optional[int] = None):
        for player, player_regret_incrs in (
            enumerate(self._regret_increments)
            if self.simultaneous
            else [(updating_player, self._regret_increments[updating_player])]
        ):
            for infostate, regret_incrs in player_regret_incrs.items():
                avg_regret = sum(regret_incrs.values()) / len(regret_incrs)
                regret_minimizer = self.regret_minimizer(infostate)
                player_policy = regret_minimizer.recommend(self.iteration)
                policy_numerator = self._avg_policy_at(
                    player, infostate, numerator=True
                )
                policy_denominator = self._avg_policy_at(
                    player, infostate, numerator=False
                )
                reach_prob = self._reach_prob[infostate]
                exp_l1_weights = dict()
                for action, regret_incr in regret_incrs.items():
                    exp_l1 = np.exp(regret_incr - avg_regret)
                    exp_l1_weights[action] = exp_l1

                    policy_weight = exp_l1 * reach_prob
                    policy_numerator[action] += policy_weight * player_policy[action]
                    policy_denominator[action] += policy_weight

                regret_minimizer.observe_regret(
                    self.iteration, lambda a: exp_l1_weights[a] * regret_incrs[a]
                )
                # delete the content of the temporary regret incr table
                for action in regret_incrs.keys():
                    regret_incrs[action] = 0.0
